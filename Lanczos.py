from numpy import zeros, sum, sqrt, abs, max, dot, where, copy
from numpy.linalg import eigh
from numpy.random import random as rand

TOL = 1e-10

def _norm_(x):
	return sqrt( sum(x**2) )

def lanczos(n, m, av, v0=[1]):
	''' n: the dimension of the matrix
		m: the number of eigenvalues to be calculated
		av: the function that performs the calculation A*v '''
	
	vs = zeros((m+1,n))
	w = zeros(n)
	alpha = zeros(m+1)
	beta = zeros(m+1)
  
	if len(v0)!=n:		# test if the initial v is given
		vs[1,:] = rand(n)
		vs[1,:] /= _norm_( vs[1,:] )
	else:
		v[1,:] = v0

	for it in range(1,m):	
		w = av(vs[it,:])
		alpha[it] = dot( vs[it,:], w)
		w = w - alpha[it] * vs[it,:] - beta[it-1] * vs[it-1,:]
		
		beta[it] = _norm_(w)
		vs[it+1,:] = w / beta[it]
				
		q = dot(vs[:it,:], vs[it+1,:])		# reorthogonalize
		vs[it+1,:] = vs[it+1,:] - dot( q, vs[:it])
		norm = _norm_(vs[it+1,:])
		vs[it+1,:] = vs[it+1,:] / norm
		
		if max(abs(q))>TOL:		# test orthogonality
			print it, "Not orthogonal.", max(abs(q))
			norm = 0.0
			num_it = 0
			while norm < 0.01 and num_it<10000:
				w = rand(n)
				w /= _norm_(w)
				w = w - dot( dot(vs[:it,:], w), vs[:it,:] )
				norm = _norm_(w)
				num_it += 1
			if num_it==10000:
				print "Maximum iteration reached."
			vs[it+1] = w / norm
			
	w = av(vs[m,:])
	alpha[m] = dot( w, vs[m,:] )			
	
	Tmm = zeros((m,m))		# construct the tridiagonal matrix
	for j in range(m-1):
		Tmm[j,j+1] = beta[j+2]
		Tmm[j+1,j] = beta[j+2]
	for j in range(m):
		Tmm[j,j] = alpha[j+1]
	
	vals, U = eigh(Tmm)
	vecs = dot( U, U )
	
	return vals, vecs
	
A = rand((1000,1000))
A = A + A.T
D0, U0 = eigh(A)

def av_test(x):
	return dot(A, x)

D, U = lanczos(1000, 100, av_test)

print D0[:3]
print D[:3]

