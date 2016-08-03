from numpy import *
from numpy.linalg import eigh
import matplotlib.pyplot as plt

def H_SSH(L, t=1.0, d=0.0):
	''' Returns the Hamiltonian of the Su-Schrieffer-Heeger (SSH) model
			H = -(t+d) C1^d C2 - (t-d) C2^d C3 - (t+d) C3^d C4 - ... + h.c. 
		In the basis [C1 C2 C3 ...] it has the form
			H =   0   -(t+d)    0      0   ...
			   -(t+d)    0   -(t-d)    0   ...
				  0   -(t-d)    0   -(t+d) ... 
				  0      0   -(t+d)    0   ...
				 ...    ...    ...    ...       '''
	
	H = zeros((L,L))
	for i in range(L-1)[::2]:
		H[i,i+1] = -(t+d)
	for i in range(L-1)[1::2]:
		H[i,i+1] = -(t-d)
	return H + H.T

def plot_phase(L=100, n=50):
	data = zeros((n,L))
	d_val = linspace(-1,1,n, endpoint=True)
	for i, d in enumerate(d_val):
		H = H_SSH(L, t=1.0, d=d)
		data[i,:], D = eigh(H)
	plt.figure("SSH spectrum")
	plt.plot(d_val, data, 'k-')
	
def plot_state(L=200, t=1.0, d=-0.1, ene=0.0):
	''' plot the state with energy within range ene-1e-6'''
	H = H_SSH(L, t=1.0, d=d)
	U, D = eigh(H)
	args = argwhere( abs(U-ene)<0.001 )
	x = arange(L)+1
	plt.figure("SSH state")
	for i, arg in enumerate(args):
		plt.plot(x, D[:,arg[0]], '-', label="E=%.2f"%U[arg])
	plt.legend(loc='best')

plot_phase()
plot_state()

plt.show()