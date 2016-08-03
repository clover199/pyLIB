import numpy as np
from scipy.sparse import kron as skron
from scipy.sparse import coo_matrix as coo
from scipy.sparse import csr_matrix as csr

TOL = 1e-10

class quantum_operator:
	""" Matrix with symmetries
		used to generate the full Hamiltonian of interaction systems """

	# initialize as an empty operator
	def __init__(self, sym=2, dim=1, datatype=np.float, fparity=None, sym_sum=None):

		def sym_sum_n(a, b):
			return (a+b+sym)%sym

		self.sym = sym		# number of total symmetry sectors
		self.dim = dim		# dimension of each symmetry sector
		self.datatype = datatype

		if fparity is None: self.fparity = np.zeros(sym, dtype=np.int)		# fermion parity
		else: self.fparity = fparity

		if sym_sum is None: self.sym_sum = sym_sum_n		# the rule to sum the symmetries
		else: self.sym_sum = sym_sum

		self.val = [[csr((dim,dim), dtype=datatype) for x in range(sym)] for y in range(sym)]
		self.basis = None
		self.L = 1


	def _empty_(self, row_sym, col_sym):
		x =np.sum( np.abs(self.val[row_sym][col_sym].data) )
		if x<TOL: return True
		else: return False


	def identity(self):
		id = quantum_operator(self.sym, self.dim, self.datatype, self.fparity, self.sym_sum)
		for i in range(self.sym):
			id.val[i][i] = csr( np.diag(np.ones(self.dim, dtype=self.datatype)) )
		return id


	def copy(self):
		sym = self.sym
		A = quantum_operator(self.sym, self.dim, self.datatype, self.fparity, self.sym_sum)
		for row in range(sym):
			for col in range(sym):
				if not self._empty_(row, col):
					A.val[row][col] = self.val[row][col].copy()
		A.basis = self.basis
		A.L = self.L
		return A


	def kron(self, B):
		fp = self.fparity	# def fp just for convenience
		ss = self.sym_sum	# def ss just for convenience

		if self.sym!=B.sym:
			raise Exception('A, B symmetries do not match.')
		sym = B.sym
		dim = sym * self.dim * B.dim
		if not np.array_equal(self.fparity, B.fparity):
			raise Exception('A, B fermion types do not match.')
		datatype = self.datatype
		if np.dtype(datatype) < np.dtype(B.datatype): datatype = B.datatype

		ret = quantum_operator(sym, dim, datatype, fp, ss)
		for row in range(sym):
			for col in range(sym):
				for row1 in range(sym):
					for col1 in range(sym):
						row2 = ss(row, -row1)
						col2 = ss(col, -col1)
						if not self._empty_(row1, col1) and not B._empty_(row2, col2):
							sign = 1 - 2*( fp[col1] * fp[ss(col2, -row2)] )
							block = self.dim * B.dim
							temp = coo( skron(self.val[row1][col1], B.val[row2][col2]) * sign )
							add = coo((temp.data, (temp.row + row1*block, temp.col + col1*block)), (dim, dim), dtype=datatype)
							ret.val[row][col] += add
		return ret


	def plus(self, B):
		if self.sym!=B.sym:
			raise Exception('A, B symmetries do not match.')
		sym = B.sym
		if self.dim!=B.dim:
			raise Exception('A, B dimensions do not match.')
		if not np.array_equal(self.fparity, B.fparity):
			raise Exception('A, B fermion types do not match.')
		if np.dtype(self.datatype) < np.dtype(B.datatype): self.astype(B.datatype)

		for row in range(sym):
			for col in range(sym):
				self.val[row][col] += B.val[row][col]
		return self


	def times(self, x):
		if np.dtype(self.datatype) < np.dtype(type(x)): self.astype(type(x))
		for row in range(self.sym):
			for col in range(self.sym):
				self.val[row][col] = self.val[row][col] * x
		return self


	def mult(self, B):
		if self.sym!=B.sym:
			raise Exception('A, B symmetries do not match.')
		sym = B.sym
		if self.dim!=B.dim:
			raise Exception('A, B dimensions do not match.')
		dim = B.dim
		if not np.array_equal(self.fparity, B.fparity):
			raise Exception('A, B fermion types do not match.')
		datatype = self.datatype
		if np.dtype(datatype) < np.dtype(B.datatype): datatype = B.datatype

		ret = quantum_operator(sym, dim, datatype, B.fparity, B.sym_sum)
		for row in range(sym):
			for col in range(sym):
				for k in range(sym):
					ret.val[row][col] += self.val[row][k].dot( B.val[k][col] )
		return ret


	def transpose(self):
		ret = quantum_operator(self.sym, self.dim, self.datatype, self.fparity, self.sym_sum)
		sym = self.sym
		for row in range(sym):
			for col in range(sym):
				if not self._empty_(row, col):
					ret.val[col][row] = self.val[row][col].T
		return ret


	def hermitian(self):
		ret = quantum_operator(self.sym, self.dim, self.datatype, self.fparity, self.sym_sum)
		sym = self.sym
		for row in range(sym):
			for col in range(sym):
				if not self._empty_(row, col):
					ret.val[col][row] = self.val[row][col].T.conjugate()
		return ret


	def todense(self):
		sym = self.sym
		dim = self.dim
		ret = np.zeros((sym*dim, sym*dim), dtype=self.datatype)
		for row in range(sym):
			for col in range(sym):
				if not self._empty_(row, col):
					ret[row*dim:(row+1)*dim, col*dim:(col+1)*dim] = self.val[row][col].todense()
		return ret


	def astype(self, dtype):
		self.datatype = dtype
		sym = self.sym
		for row in range(sym):
			for col in range(sym):
				if not self._empty_(row, col):
					self.val[row][col] = self.val[row][col].astype(dtype)
		return self



def basis(n):
	""" # the basis array has the form basis[sector, state, L] """
	return np.arange(n, dtype=int).reshape(n,1,1)

def spinful_fermion_basis():
	return np.array([ [[0],[3]], [[1],[2]] ], dtype=int)


def sigma_x():
	Sx = quantum_operator(2, 1)
	Sx.val[1][0] = csr( np.array([[1.]]) )
	Sx.val[0][1] = csr( np.array([[1.]]) )
	Sx.basis = basis(2)
	return Sx

def sigma_y():
	Sy = quantum_operator(2, 1, datatype=np.complex)
	Sy.val[0][1] = csr( np.array([[-1.j]]) )
	Sy.val[1][0] = csr( np.array([[1.j]]) )
	Sy.basis = basis(2)
	return Sy

def sigma_z():
	Sz = quantum_operator(2, 1)
	Sz.val[0][0] = csr( np.array([[1.]]) )
	Sz.val[1][1] = csr( np.array([[-1.]]) )
	Sz.basis = basis(2)
	return Sz

def tau_n(n=3):
	w = np.cos(2*np.pi/n) + 1.j * np.sin(2*np.pi/n)
	tau = quantum_operator(n, 1, datatype=np.complex)
	for i in range(n):
		tau.val[i][i] = csr( np.array([[w**i]]) )
	tau.basis = basis(n)
	return tau

def sigma_n(n=3):
	sigma = quantum_operator(n, 1)
	for i in range(n):
		sigma.val[i][(i+n+1)%n] = csr( np.array([[1.]]) )
	sigma.basis = basis(n)
	return sigma

def spin1_x():
	Sx = quantum_operator(3, 1)
	Sx.val[0][1] = csr( np.array([[1.]]) )
	Sx.val[1][0] = csr( np.array([[1.]]) )
	Sx.val[1][2] = csr( np.array([[1.]]) )
	Sx.val[2][1] = csr( np.array([[1.]]) )
	Sx.times(1./np.sqrt(2))
	Sx.basis = basis(3)
	return Sx

def spin1_y():
	Sy = quantum_operator(3, 1)
	Sy.val[0][1] = csr( np.array([[-1.]]) )
	Sy.val[1][0] = csr( np.array([[1.]]) )
	Sy.val[1][2] = csr( np.array([[-1.]]) )
	Sy.val[2][1] = csr( np.array([[1.]]) )
	Sy.times(1.j/np.sqrt(2))
	Sy.basis = basis(3)
	return Sy

def spin1_z():
	Sz = quantum_operator(3, 1)
	Sz.val[0][0] = csr( np.array([[1.]]) )
	Sz.val[2][2] = csr( np.array([[-1.]]) )
	Sz.basis = basis(3)
	return Sz


def fermion_c():
	f = quantum_operator(2, 1, fparity=np.array([0,1], dtype=np.int))
	f.val[0][1] = csr( np.array([[1.]]) )
	f.basis = basis(2)
	return f

def fermion_up():
	f = quantum_operator(2, 2, fparity=np.array([0,1], dtype=np.int))
	f.val[0][1] = csr( np.array([[1.,0.],[0.,0.]]) )
	f.val[1][0] = csr( np.array([[0.,0.],[0.,1.]]) )
	f.basis = spinful_fermion_basis()
	return f

def fermion_down():
	f = quantum_operator(2, 2, fparity=np.array([0,1], dtype=np.int))
	f.val[0][1] = csr( np.array([[0.,1.],[0.,0.]]) )
	f.val[1][0] = csr( np.array([[0.,-1.],[0.,0.]]) )
	f.basis = spinful_fermion_basis()
	return f

def sym_sum4(a, b):
	pp = np.array([	[0,1,2,3],
					[1,0,3,2],
					[2,3,0,1],
					[3,2,1,0]])
	if b<0: return pp[a,-b]
	return pp[a,b]

def fermion_up4():
	f = quantum_operator(4, 1, fparity=np.array([0,1,1,0], dtype=np.int), sym_sum=sym_sum4)
	f.val[0][1] = csr( np.array([[1.]]) )
	f.val[2][3] = csr( np.array([[1.]]) )
	f.basis = basis(4)
	return f

def fermion_down4():
	f = quantum_operator(4, 1, fparity=np.array([0,1,1,0], dtype=np.int), sym_sum=sym_sum4)
	f.val[0][2] = csr( np.array([[1.]]) )
	f.val[1][3] = csr( np.array([[-1.]]) )
	f.basis = basis(4)
	return f
