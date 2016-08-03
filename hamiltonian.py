import numpy as np
import operators as op

print_p = False

def basis(L, b1, ss=None):
	''' -> L: system size
		-> b1: the basis of one site
		-> ss: the summation rule for the symmetries '''

	if len(b1.shape)!=3:
		raise Exception("The basis for one site should have shape (*,*,*)")

	sym = b1.shape[0]
	dim = b1.shape[1]

	if ss is None:
		def ss(a, b):
			return (a+b+sym)%sym

	print "\n--> Construct the basis for system L=%d with base shape sym=%d, dim=%d..." % (L, sym, dim)

	store = [ np.arange(0, dtype=int).reshape(sym,0,0) for i in range(L+1) ]
	store[1] = b1

	def iterate(l):
		if l==0: return store[0]

		if store[l].shape[1] != 0: return store[l]

		le = int(l/2)
		lb = iterate(le)
		rb = iterate(l-le)

		if le==0: return rb
		if le==l: return lb

		store[l] = np.zeros((sym, np.power(sym*dim,l)/sym, l), dtype=int)
		# the basis array has the form (sym=3, dim=1, L=3):
		#  000  001  002
		#  012  010  011
		#  021  022  020
		#  ...  ...  ...
		# basis[sector, state, L]

		for s in range(sym):
			for s1 in range(sym):
				s2 = ss(s, -s1)
				for i1 in range(lb.shape[1]):
					for i2 in range(rb.shape[1]):
						index = s1 * lb.shape[1] * rb.shape[1] + i1 * rb.shape[1] + i2
						store[l][s, index, :le] = lb[s1, i1, :]
						store[l][s, index, le:] = rb[s2, i2, :]
		return store[l]

	print "--> Basis constructed."
	return iterate(L)


def operator(L, ope, n):
	if len(ope) != len(n):
		raise Exception("The size of the operators and locations are not the same")
	if min(n)<0 or max(n)>=L:
		raise Exception("The position of the operator should be in range 0~%d" % L-1)

	if print_p: print "\n--> Construct the operator for system of size L=%d ..." % (L)

	id = ope[0].identity()
	store = [ id for x in range(L) ]
	for i,x in enumerate(n): store[x] = ope[i]

	def iterate(a,b):
		if a==b: return store[a]

		ab = a + int( (b-a+1)/2 )
		lH = iterate(a, ab-1)
		rH = iterate(ab, b)

		return lH.kron( rH )

	H = iterate(0,L-1)
	if print_p: print "--> Operator constructed."
	H.basis = id.basis
	H.L = L
	return H


def Potts_n(L, n=3, J=1., h=1., theta=0., phi=0., pbc=False):
	''' - J * Sz * Sz * e^{i theta} + h.c. - h * Sx * e^{i phi}'''

	print "\n--> Construct the Hamiltonian for the %d-Potts model of size L=%d ..." % (n, L)

	store = [ [None, None, None] for i in range(L+1) ]
	store[1][0] = op.tau_n(n).times( -h/2 * (np.cos(phi) + 1j * np.sin(phi)) )
	store[1][1] = op.sigma_n(n)
	store[1][2] = op.sigma_n(n).hermitian().times( -J * (np.cos(theta) + 1j * np.sin(theta)) )

	def iterate(l):
		if store[l] != store[0]: return store[l]

		le = int(l/2)
		lH, llo, lro = iterate(le)
		rH, rlo, rro = iterate(l-le)

		lid = lH.identity()
		rid = rH.identity()
		store[l][0] = lH.kron( rid )
		temp = lid.kron( rH )
		store[l][0].plus(temp)
		store[l][0].plus( lro.kron(rlo) )
		store[l][1] = llo.kron(rid)
		store[l][2] = lid.kron(rro)
		return store[l]

	H, lo, ro = iterate(L)
	if pbc:
		edge = operator(L, [ store[1][1], store[1][2] ], [0,L-1])
		edge.times(-J)
		H.plus(edge)
	temp = H.hermitian()
	H.plus(temp)
	print "--> Hamiltonian constructed."
	H.basis = op.basis(n)
	H.L = L
	return H


def AKLT(L, pbc=False):
	'''  S1 * S1 + 1/3 (S1 * S1)^2 '''

	print "\n--> Construct the Hamiltonian for the AKLT model of size L=%d ..." % L

	store = [ [None, None, None, None, None, None, None] for i in range(L+1) ]
	store[1][0] = op.spin1_x().times(0)
	store[1][1] = op.spin1_x()
	store[1][2] = op.spin1_x()
	store[1][3] = op.spin1_y()
	store[1][4] = op.spin1_y()
	store[1][5] = op.spin1_z()
	store[1][6] = op.spin1_z()

	def iterate(l):
		if store[l] != store[0]: return store[l]

		le = int(l/2)
		lH, llx, lrx, lly, lry, llz, lrz = iterate(le)
		rH, rlx, rrx, rly, rry, rlz, rrz = iterate(l-le)

		lid = lH.identity()
		rid = rH.identity()
		store[l][0] = lH.kron( rid )
		temp = lid.kron( rH )
		store[l][0].plus(temp)
		temp = lrx.kron(rlx)
		temp.plus( lry.kron(rly) )
		temp.plus( lrz.kron(rlz) )
		store[l][0].plus(temp)
		store[l][0].plus( temp.mult(temp).times(1/3.) )
		store[l][1] = llx.kron(rid)
		store[l][2] = lid.kron(rrx)
		store[l][3] = lly.kron(rid)
		store[l][4] = lid.kron(rry)
		store[l][5] = llz.kron(rid)
		store[l][6] = lid.kron(rrz)
		return store[l]

	H, lx, rx, ly, ry, lz, rz = iterate(L)
	if pbc:
		edge = lx.mult(rx)
		edge.plus( ly.mult(ry) )
		edge.plus( lz.mult(rz) )
		H.plus(edge)
		H.plus( edge.mult(edge).times(1./3) )
	print "--> Hamiltonian constructed."
	H.basis = op.basis(3)
	H.L = L
	return H


def Kitaev(L, t=1., u=1., d=1., pbc=False):
	''' - t * c^d * c + 2u * n + d * c * c
		= ( -t * c^d + d * c) * c + h.c. + 2u * n '''
	print "\n--> Construct the Hamiltonian for the Kitaev model of size L=%d ..." % (L)

	store = [ [None, None, None] for i in range(L+1) ]
	store[1][0] = op.fermion_c().hermitian().mult( op.fermion_c() ).times( u )
	store[1][1] = op.fermion_c().hermitian().times(-t).plus( op.fermion_c().times(d) )
	store[1][2] = op.fermion_c()

	def iterate(l):
		if store[l] != store[0]: return store[l]

		le = int(l/2)
		lH, llo, lro = iterate(le)
		rH, rlo, rro = iterate(l-le)

		lid = lH.identity()
		rid = rH.identity()
		store[l][0] = lH.kron( rid )
		temp = lid.kron( rH )
		store[l][0].plus(temp)
		store[l][0].plus( lro.kron(rlo) )
		store[l][1] = llo.kron(rid)
		store[l][2] = lid.kron(rro)
		return store[l]

	H, lo, ro = iterate(L)
	if pbc:
		edge = operator(L, [ store[1][1], store[1][2] ], [0,L-1])
		edge.times(-d)
		H.plus(edge)
	temp = H.hermitian()
	H.plus(temp)
	print "--> Hamiltonian constructed."
	H.basis = op.basis(2)
	H.L = L
	return H


def SSH(L, t=1., d=1., pbc=False):
	''' - (t-d) * c^d * c - (t+d) * c^d * c '''

	print "\n--> Construct the Hamiltonian for the SSH model of size L=%d ..." % (L)

	ope_z = op.fermion_c().times(0.0)
	ope_c = op.fermion_c()
	ope_cd = op.fermion_c().hermitian()

	def iterate(a,b):

		if a==b: return [ope_z, ope_c, ope_cd]

		ab = a + int( (b-a+1)/2 )
		lH, llo, lro = iterate(a, ab-1)
		rH, rlo, rro = iterate(ab, b)

		lid = lH.identity()
		rid = rH.identity()
		H = lH.kron( rid )
		temp = lid.kron( rH )
		H.plus(temp)
		td = -t+d
		if ab%2: td = -t-d
		H.plus( lro.kron(rlo).times(td) )
		return [H, llo.kron(rid), lid.kron(rro)]

	H, lo, ro = iterate(1,L)
	if pbc:
		edge = operator(L, [ope_cd, ope_c], [0,L-1])
		edge.times(t+d)
		H.plus(edge)
	temp = H.hermitian()
	H.plus(temp)
	print "--> Hamiltonian constructed."
	H.basis = op.basis(2)
	H.L = L
	return H


def Hubbard(L, t=1., u=1., f=1., pbc=False):
	''' - t c^d c + u n + f n n  '''
	print "\n--> Construct the Hamiltonian for the Hubbard model of size L=%d ..." % (L)

	n_up = op.fermion_up().hermitian().mult( op.fermion_up() )
	n_down = op.fermion_down().hermitian().mult( op.fermion_down() )
	nn = n_up.mult( n_down ).times( f/2. )
	mH = nn.plus( n_up.plus(n_down).times(u/2.) )
	lou = op.fermion_up().hermitian().times(-t)
	lod = op.fermion_down().hermitian().times(-t)

	def iterate(a, b):
		if a==b:
			return [mH, lou, op.fermion_up(), lod, op.fermion_down()]

		ab = a + int( (b-a+1)/2 )
		lH, llou, lrou, llod, lrod = iterate(a, ab-1)
		rH, rlou, rrou, rlod, rrod = iterate(ab, b)

		lid = lH.identity()
		rid = rH.identity()
		H0 = lH.kron( rid )
		temp = lid.kron( rH )
		H0.plus(temp)
		H0.plus( lrou.kron(rlou) )
		H0.plus( lrod.kron(rlod) )
		return [H0, llou.kron(rid), lid.kron(rrou), llod.kron(rid), lid.kron(rrod)]

	H, Hlou, Hrou, Hlod, Hrod= iterate(1,L)
	if pbc:
		edge = operator(L, [ lou, op.fermion_up() ], [0,L-1])
		edge.times(t)
		H.plus(edge)
		edge = operator(L, [ lod, op.fermion_down() ], [0,L-1])
		edge.times(t)
		H.plus(edge)
	temp = H.hermitian()
	H.plus(temp)
	print "--> Hamiltonian constructed."
	H.basis = op.spinful_fermion_basis()
	H.L = L
	return H


def Peierls_Hubbard(L, t=1., dt=1., U=1., pbc=False):
	''' - (t+-dt) c^d c +  U (n-1/2) (n-1/2)  '''
	print "\n--> Construct the Hamiltonian for the Peierls_Hubbard model of size L=%d ..." % (L)

	n_up = op.fermion_up().hermitian().mult( op.fermion_up() )
	n_down = op.fermion_down().hermitian().mult( op.fermion_down() )
	mH = n_up.mult( n_down ).times( U/2. )
	lou = op.fermion_up().hermitian()
	lod = op.fermion_down().hermitian()

	def iterate(a, b):
		if a==b:
			return [mH, lou, op.fermion_up(), lod, op.fermion_down()]

		ab = a + int( (b-a+1)/2 )
		td = -t+dt
		if ab%2: td = -t-dt
		lH, llou, lrou, llod, lrod = iterate(a, ab-1)
		rH, rlou, rrou, rlod, rrod = iterate(ab, b)

		lid = lH.identity()
		rid = rH.identity()
		H0 = lH.kron( rid )
		temp = lid.kron( rH )
		H0.plus(temp)
		H0.plus( lrou.kron(rlou).times(td) )
		H0.plus( lrod.kron(rlod).times(td) )
		return [H0, llou.kron(rid), lid.kron(rrou), llod.kron(rid), lid.kron(rrod)]

	H, Hlou, Hrou, Hlod, Hrod= iterate(1,L)
	if pbc:
		edge = operator(L, [ lou, op.fermion_up() ], [0,L-1])
		edge.times(-t-dt)
		H.plus(edge)
		edge = operator(L, [ lod, op.fermion_down() ], [0,L-1])
		edge.times(-t-dt)
		H.plus(edge)
	temp = H.hermitian()
	H.plus(temp)
	print "--> Hamiltonian constructed."
	H.basis = op.spinful_fermion_basis()
	H.L = L
	return H


def number(L):
	''' n1 + n2 + ...  '''
	print "\n--> Construct the number operator for the spinful system of size L=%d ..." % (L)

	n_up = op.fermion_up().hermitian().mult( op.fermion_up() )
	n_down = op.fermion_down().hermitian().mult( op.fermion_down() )
	mH = n_up.plus( n_down )
	H = operator(L, [mH for i in range(L)], range(L))
	print "--> operator constructed."
	H.basis = op.spinful_fermion_basis()
	H.L = L
	return H


# Free fermion Hamiltonians

def Kitaev_free(L, t=1., u=1., d=1., pbc=False):
	''' u * c^d * c - u * c * c^d
		- t * c^d * c + t * c * c^d
	 	+ d * c * c  - d * c^d * c*d '''

	H = np.zeros((L*2,L*2))
	index = np.arange(L*2)
	H[index[0:2*L:2], index[0:2*L:2]] = u/2.
	H[index[1:2*L:2], index[1:2*L:2]] = -u/2.
	H[index[0:2*L-2:2], index[2:2*L:2]] = -t/2.
	H[index[1:2*L-2:2], index[3:2*L:2]] = t/2.
	H[index[0:2*L-2:2], index[3:2*L:2]] = -d/2.
	H[index[1:2*L-2:2], index[2:2*L:2]] = d/2.
	if pbc:
		H[0, 2*L-2] = -t/2
		H[0, 2*L-1] = -d/2
		H[1, 2*L-2] = d/2
		H[1, 2*L-1] = t/2
	return H + H.T


def SSH_free(L, t=1., d=1., pbc=False):
	''' - (t-d) * c^d * c - (t+d) * c^d * c '''

	H = np.zeros((L,L))
	index = np.arange(L)
	H[index[0:L-1:2], index[1:L:2]] = -(t-d)
	H[index[1:L-2:2], index[2:L-1:2]] = -(t+d)
	if pbc:
		H[0,L-1] = -(t+d)
	return H + H.T
