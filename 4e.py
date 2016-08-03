import numpy as np
import functions as func
import hamiltonian as ham
import operators as op

import scipy.sparse as sparse
from scipy.sparse import csr_matrix as csr
from scipy.sparse import coo_matrix as coo
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh, svd

def FK(L=8, sec=0, t1=1., t2=1.0, f0=1., f1=1., pbc=False):
	''' see the paper for Hamiltonian
		hopping t: i c^d c
		f0: -(2n-1)(2n-1)
		f1: -4(n+n-1)(n+n-1) + 8(cccc+h.c.) '''
	if L not in [1,2,4,8]:
		raise Exception("Illegal L value")
	print "\n--> Construct the Hamiltonian for the FK model of size L=%d ..." % (L)

	n_up = op.fermion_up4().hermitian().mult( op.fermion_up4() )
	n_down = op.fermion_down4().hermitian().mult( op.fermion_down4() )
	pp = op.fermion_up4().mult( op.fermion_down4() )	# cc
	ide = n_up.identity()
	n2_up = n_up.copy().times(-2).plus( ide )		# 1-2n
	n2_down = n_down.copy().times(-2).plus( ide )	# 1-2n
	nn = n2_up.copy().plus(n2_down)		# 2(1-n-n)
	mH = n2_up.mult( n2_down ).times(-f0/2.)
	if L==1:
		H = mH.hermitian().plus(mH)
	 	H.L = 1
	 	H.basis = op.basis(4)
		return H

	# the 2-stie ham with the hopping t1 and onsite-interaction f0
	H2 = op.fermion_up4().hermitian().kron( op.fermion_up4() )
	H2.plus( op.fermion_down4().hermitian().kron( op.fermion_down4() ) )
	H2.times( 2.j*float(t1) )
	H2.plus( mH.kron(ide).plus( ide.kron(mH) ) )
	if L==2:
		H = H2.hermitian().plus(H2)
	 	H.L = 1
	 	H.basis = op.basis(4)
		return H

	# the 4 site ham with interaction f1 and hopping t2
	H4 = ham.operator(4, [op.fermion_up4().hermitian(), op.fermion_up4()], [1,3])
	H4.plus( ham.operator(4, [op.fermion_down4().hermitian(), op.fermion_down4()], [1,3]) )
	H4.times( 2.j*float(t2) )
	ide = H2.identity()
	H4.plus( H2.kron(ide).plus( ide.kron(H2) ) )
	H4.plus( ham.operator(4, [nn, nn], [1,2]).times(-f1/2.) )
	H4.plus( ham.operator(4, [pp, pp], [1,2]).times(f1*8.) )

	if L==4:
		H = H4.hermitian().plus(H4)
	 	H.L = 1
	 	H.basis = op.basis(4)
		return H

	def kron_s(A, B, sec):
		fp = A.fparity
		ss = A.sym_sum
		sym = A.sym
		dim = sym * A.dim * B.dim
		ret = csr((dim, dim), dtype=np.complex)
		for row1 in range(sym):
			for col1 in range(sym):
				row2 = ss(sec, -row1)
				col2 = ss(sec, -col1)
				if not A._empty_(row1, col1) and not B._empty_(row2, col2):
					sign = 1 - 2*( fp[col1] * fp[ss(col2, -row2)] )
					temp = coo( sparse.kron(A.val[row1][col1], B.val[row2][col2]) * sign )
					block = A.dim * B.dim
					add = coo((temp.data, (temp.row + row1*block, temp.col + col1*block)),(dim, dim), dtype=np.complex)
					ret += add
		return ret

	# the 8 site ham
	lup = ham.operator(4, [op.fermion_up4().hermitian()], [2])
	rup = ham.operator(4, [op.fermion_up4()], [0])
	ldown = ham.operator(4, [op.fermion_down4().hermitian()], [2])
	rdown = ham.operator(4, [op.fermion_down4()], [0])
	ide = H4.identity()
	ln = ham.operator(4, [nn], [3])
	rn = ham.operator(4, [nn], [0])
	lp = ham.operator(4, [pp], [3])
	rp = ham.operator(4, [pp], [0])

	Hs = coo((16384, 16384), dtype=np.complex)
	Hs += kron_s(lup, rup, sec)
	Hs += kron_s(ldown, rdown, sec)
	Hs *= 2.j*float(t2)
	Hs += kron_s(H4, ide, sec)
	Hs += kron_s(ide, H4, sec)
	Hs += kron_s(ln, rn, sec)*(-f1/2.)
	Hs += kron_s(lp, rp, sec)*(f1*8.)

	if pbc:
		edge = kron_s(rup,lup, sec)
		edge += kron_s(rdown, ldown, sec)
		edge *= -2.j*float(t2)
		Hs += edge
		Hs += kron_s(rn, ln, sec)*(-f1/2.)
		Hs += kron_s(rp, lp, sec)*(f1*8.)

	Hs += Hs.conjugate().transpose()
	print "--> Hamiltonian constructed."
	return Hs

def FK_eig(L=8, t1=1., t2=1., f0=1., f1=1., pbc=False):
	evals_all = []
	evecs_all = []

	if L==8:
		for sec in range(0,4):
			H = FK(L=8, sec=sec, t1=t1, t2=t2, f0=f0, f1=f1, pbc=pbc)
			evals, evecs = eigsh(H, which='SA')
			arg = np.argsort(evals)
			evals_all.append( evals[arg] )
			evecs_all.append( evecs[:,arg] )
		return np.array(evals_all), np.array(evecs_all)

	H = FK(L=L, sec=0, t1=t1, t2=t2, f0=f0, f1=f1, pbc=pbc)
	for sec in range(0,4):
	 	D, U = eigh(H.val[sec][sec].todense())
		evals_all.append(D)
		evecs_all.append(U)
	return np.array(evals_all), np.array(evecs_all)

L = 8
basis = ham.basis(L, op.basis(4), op.fermion_up4().sym_sum)
evals, evecs = FK_eig(L=L, t1=1.0, t2=1., f0=0.25, f1=0.25, pbc=True)
print evals[:,:5]

sec = 0
rho = func.reduced_density_matrix(evecs[sec,:,0], basis[sec], 4, 4)
u, d, v = svd(rho)
ee = -np.sum(d**2 * np.log( d**2) )
print "Entanglement entropy:", ee, "or", ee/np.log(2), "ln2"
rho = func.reduced_density_matrix(evecs[sec,:,1], basis[sec], 4, 4)
u, d, v = svd(rho)
ee = -np.sum(d**2 * np.log( d**2) )
print "Entanglement entropy:", ee, "or", ee/np.log(2), "ln2"
sec = 3
rho = func.reduced_density_matrix(evecs[sec,:,0], basis[sec], 4, 4)
u, d, v = svd(rho)
ee = -np.sum(d**2 * np.log( d**2) )
print "Entanglement entropy:", ee, "or", ee/np.log(2), "ln2"
rho = func.reduced_density_matrix(evecs[sec,:,1], basis[sec], 4, 4)
u, d, v = svd(rho)
ee = -np.sum(d**2 * np.log( d**2) )
print "Entanglement entropy:", ee, "or", ee/np.log(2), "ln2"
