import numpy as np
from numpy.linalg import eigh, svd

from hamiltonian import basis as ham_basis
from hamiltonian import operator as ham_ope
from operators import TOL

def convert(A, base=2):
	''' Convert m base-radix numbers to decimal numbers.
		-> A: (m,n)-array. Each row is a base-radix number of size n.
		-> base: default 2. The radix of the input number.
		Return: (m)-array of decimal numbers. '''

	A = np.array(A)
	return A.dot( np.power(base, np.arange(A.shape[1]-1,-1,-1)) ).astype(int)


def reduced_density_matrix(state, basis, l, dim):
	''' Reshape the state into a a*b matrix so that a/b represents only left/right sub-system
		-> state: (n)-array representing the state
		-> basis: (n,L)-array. Each row represents a product state
		-> l: the size of the left system
		-> dim: the local degree of freedom of one site
		Return: the matrix of the reshaped the state '''

	state = np.array(state)
	if state.shape[0] != basis.shape[0]:
		raise Exception('Sizes of state and basis do not match.')
	L = basis.shape[1]

	left = convert(basis[:,:l], dim)
	right = convert(basis[:,l:], dim)
	state_new = np.zeros(( np.power(dim, l), np.power(dim, L-l) ), dtype=state.dtype)
	for i in range(state_new.shape[0]):
		arg = np.where(left==i)
		state_new[i, right[arg] ] = state[arg]
	return state_new


def discrete_reduced_density_matrix(state, basis, sub, dim):
	''' Reshape the state into a a*b matrix so that a/b represents only left/right sub-system
		-> state: (n)-array representing the state
		-> basis: (n,L)-array. Each row represents a product state
		-> sub: (m,2)-array giving the subsystem range NOT to be traced off
		-> dim: the local degree of freedom of one site
		Return: the matrix of the reshaped the state '''

	state = np.array(state)
	if state.shape[0] != basis.shape[0]:
		raise Exception('Sizes of state and basis do not match.')
	L = basis.shape[1]					# the total system size
	sub = np.array(sub)
	pos = np.zeros(0, dtype=int)		# get the remaining sites
	for i in range(sub.shape[0]):
		pos = np.concatenate((pos, np.arange(sub[i,1]-sub[i,0])+sub[i,0]))
	pos = np.sort( np.unique(pos) )
	print "sites that remain:", pos
	l = pos.shape[0]					# the remaining system size
	d_tr = int(dim**(L-l))				# the dimension of the tracing sites
	d_rem = int(dim**l)					# the dimension of the remaining sites
	trans = np.zeros(L, dtype=int)		# the array used to transform the basis to index
	trans[pos] = np.power(dim, np.arange(l-1,-1,-1))
	index_rem = np.dot(basis, trans)	# the index in rho of each basis state
	arg = np.argwhere(trans==0).ravel()			# the sites to be traced off
	trans[pos] = 0
	trans[arg] = np.power(dim, np.arange(L-l-1,-1,-1))
	index_tr = np.dot(basis, trans)		# the index to be traced off
	state_new = np.zeros(dim**L, dtype=state.dtype)
	state_new[index_tr*d_rem+index_rem] = state
	return state_new.reshape((d_tr, d_rem))


def entanglement_entropy(state, l, H, s=0):
	''' return the entanglement entropy at cut l for the state
		-> state: (n)-array representing the state
		-> l: the position of the cut. Equals the size of the left system
		-> H: the Hamiltonian operator
		-> s: default 0. The symmetry sector of the state
		Return: the value of the entanglement entropy '''

	basis = ham_basis(H.L, H.basis, H.sym_sum)
	rho = reduced_density_matrix(state, basis[s], l, H.basis.ravel().shape[0])
	u, d, v = svd(rho)
	return -np.sum(d**2 * np.log( d**2) )


def fit_central_charge(ee):
	''' calculate the central charge from fitting the entanglement entropy
	-> ee: the entanglement entropy at cuts 1 ~ L-1
	Return: the fitting polynomial '''

	ee = np.array(ee)
	L = ee.shape[0]+1
	l = np.arange(1,L)
	x = np.log(np.sin(np.pi*l/L))/6
	a = 0
	b = L
	p = np.polyfit(x[a:b], ee[a:b], 1)
	print "central charge c=%.2f" % p[0]
	return p


def solve_hamiltonian(H):
	''' solve the hamitonian
		-> H: the Hamiltonian
		Return: the (sym, dim) array of eigenvalues
				and (sym, dim, dim) array of eigenvectors'''
	evals = []
	evecs = []
	for s in range(H.sym):
		print "\t calculate eigenvalues for symmetry sector %d ..." % s
		D, U = eigh(H.val[s][s].todense())
		evals.append(D)
		evecs.append(U)
	return np.array(evals), np.array(evecs)


def expectation(L, state, sec, opes):
	''' calculate the expectation values of operators
		-> L: system size
		-> sec: the symmetry sector of the state
		-> state: the state to calculate
		-> opes: array of operators
		Return: (l) array where l = L - len(opes) + 1
			each element R[i] is <opes[0] opes[1] ... > at position i, i+1 ... '''

	exp = np.zeros( L - len(opes) + 1 )
	pos = np.arange(len(opes))
	for l in range( len(exp) ):
		ope = ham_ope(L, opes, pos+l)
		exp[l] = np.dot(state, ope.val[sec][sec].dot(state) )
	return exp


def green(w, ope, L, evals, evecs):
	''' calculate the greens function G_ij(w) using the Lehmann decomposition
		-> w: the frequency
		-> ope: the particle annihilation operator
		-> L: system size
		-> evals: (sym, dim) array of all the eigenenergies
		-> evecs: (sym, dim, n) array of all the eigenstates
		Return: G_ij(w) = <g|o_i|n><n|od_j|g> / (w-En) + <g|od_i|n><n|o_j|g> / (w+En) '''

	evals = np.array(evals)
	w = np.array(w)
	sym = evals.shape[0]
	dim = evals.shape[1]
	sec = 0
	g0 = float('inf')	# the ground state energy
	for s in range(sym):
		if evals[s][0] < g0:
			sec = s
			g0 = evals[s][0]

	nl = np.zeros((dim, L))		# <n|o|g>
	nld = np.zeros((dim, L))	# <n|od|g>
	Gij = np.zeros((w.shape[0], L, L))
	for s in range(sym):
		ope_i = ham_ope(L, [ope], [0])
		if not ope_i._empty_(s, sec):
			for i in range(L):
				ope_i = ham_ope(L, [ope], [i])
				nl[:,i] = np.dot(evecs[s].T, ope_i.val[s][sec].dot(evecs[sec][:,0]) )
				ope_i = ope_i.hermitian()
				nld[:,i] = np.dot(evecs[s][:,:].T, ope_i.val[s][sec].dot(evecs[sec][:,0]) )
			for j, wj in enumerate(w):
				en = np.diag( 1./( TOL + wj - (evals[s]-evals[sec][0]) ) )
				Gij[j,:,:] += np.dot( nld.T, np.dot( en, nld ) )
				en = np.diag( 1./( TOL + wj + (evals[s]-evals[sec][0]) ) )
				Gij[j,:,:]	+= np.dot( nl.T, np.dot( en, nl ) )
	return Gij

def fourier(x):
	''' -> x: (n)-array
		return: (n)-array of the Fourier transformation of x
		y[k] = x[j] * exp( 2 pi i j k / n ) '''
	x = np.array(x)
	n = x.shape[0]
	y = np.zeros(n+1, dtype=np.complex)
	for k in range(n):
		y[k] = np.dot( x , np.exp( 1.j*(2*np.pi/n*k*np.arange(n)) ) )
	y[-1] = y[0]
	return y
	# Gk = np.zeros(L, dtype=np.complex)
	# for k in range(L):
		# Gk[k] = G(k)
	# # plt.plot(np.arange(L)*2*np.pi/L, np.real(Gk), 'o')
	# return np.real(Gk)
