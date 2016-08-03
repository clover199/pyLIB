from numpy import *
from numpy.linalg import eigh
import matplotlib.pyplot as plt

# the magnetic field B is in the unit 2*pi/a^2, so the flux within each
# lattice square is 2*pi*B.

def spectrum_QH_PBC(Lx, Ly, tx=1.0, ty=1.0,  B=1, q=3):
	''' Returns the Hamiltonian of the quantum spin Hall in Landau gauge A=(0,xB)
		with periodic boundary condition (PBC) in both x and y direction.
			H = - t C^d C exp(iA)
		For details about the Fourier transformed Hamiltonian, see Taylor's book.
		B=p/q is a rational number. We take B as p in this case '''
	if Lx%q != 0: raise Exception("q should divide Lx")
	ene = zeros((Lx/q, Ly, q))
	for kx in range(Lx/q):
		for ky in range(Ly):
			H = zeros((q,q), dtype=complex)
			for i in range(q-1):
				H[i,i+1] = -ty
			H[0, q-1] = -ty * exp(q * 2*pi/Ly*ky * 1j)
			for i in range(q):
				H[i,i] = - tx * cos( 2*pi/Lx*kx + 2*pi*B/q*i )
			H += conjugate(H.T)
			ene[kx,ky,:], U = eigh(H)
	return ene

def H_QH_OBC(Lx, Ly, tx=1.0, ty=1.0, B=1.0):
	''' Returns the Hamiltonian of the quantum Hall in Landau gauge A=(0,xB)
		with oben boundary condition (PBC) in both x and y direction.
			H = - t C^d C exp(iA) 
		in the basis [C11 C12 C13 ... C21 C22... ] '''
		
	H = zeros((Lx*Ly,Lx*Ly), dtype=complex)
	for x in range(Lx-1):
		for y in range(Ly-1):
			H[ x*Ly+y, x*Ly+y+1 ] = -ty * exp(2*pi*x*B*1j)
			H[ x*Ly+y, (x+1)*Ly+y ] = -tx
	return H + conjugate(H.T)
	
def spectrum_QH(Lx, Ly, tx=1.0, ty=1.0,  B=1./6, bc=1):
	''' Returns the Hamiltonian of the quantum spin Hall in Landau gauge A=(0,xB)
		with OBC/PBC in x direction and PBC in y direction. '''
	
	ene = zeros((Ly, Lx))
	for ky in range(Ly):
		H = zeros((Lx,Lx), dtype=complex)
		for i in range(Lx-1):
			H[i,i+1] = -tx
		H[0, Lx-1] = -tx*bc
		for i in range(Lx):
			H[i,i] = - ty * cos( 2*pi/Ly*ky + 2*pi*B*i )
		H += conjugate(H.T)
		ene[ky,:], U = eigh(H)
	return ene

def plot_spectrum(Lx=20, Ly=50, B=1, q=6):
	ene = spectrum_QH_PBC(Lx, Ly, B=B, q=q)
	plt.figure("QH spectrum")
	ky = linspace(-pi,pi, Ly, endpoint=False)
	for i in range(Lx/q):
		plt.plot(ky, ene[i,:,:], 'k-')
	plt.plot([-pi,pi],[0,0],'k--')
	plt.xlim([-pi,pi])
	plt.xticks([-pi,0,pi],[r"-$\pi$",0,r"$\pi$"])
	plt.xlabel(r"$k_y$")

def plot_spectrum_edge(Lx=50, Ly=200, B=1./6, bc=1):
	ene = spectrum_QH(Lx, Ly, B=B, bc=bc)
	plt.figure("QH spectrum " + ["with","without"][bc] + " edge states")
	ky = linspace(-pi,pi, Ly, endpoint=False)
	plt.plot(ky, ene, 'k-')
	plt.plot([-pi,pi],[0,0],'k--')
	plt.xlim([-pi,pi])
	plt.xticks([-pi,0,pi],[r"-$\pi$",0,r"$\pi$"])
	plt.xlabel(r"$k_y$")

def plot_energies(Lx=20, Ly=50):
	B_val = array([1./100, 1./10, 1./5, 1./4, 1./3, 1./2, 2./3, 3./4, 4./5, 9./10, 99./100])
	data = zeros((len(B_val),Lx*Ly))
	print "Print the zero energies for each B value:"
	for i, B in enumerate(B_val):
		H = H_QH_OBC(Lx, Ly, B=B)
		data[i,:], D = eigh(H)
		print "B=%.2f" % B, "\tzero energies:", data[i,where(abs(data[i,:])<0.001)]
	plt.figure("QH energies")
	plt.plot(B_val, data, 'k_')
	plt.xlabel("B")
	plt.xlim([0,1])
	plt.xticks(B_val,
	['1/100', '1/10', '1/5', '1/4', '1/3', '1/2', '2/3', '3/4', '4/5', '9/10', '99/100'])

def plot_energy(Lx=10, Ly=50, B=0.5):
	H = H_QH_OBC(Lx, Ly, B=B)
	D, U = eigh(H)
	plt.figure("QH energy")
	plt.plot(array([0,1]),array([D,D]), 'k-')
	
# plot_spectrum(Lx=60, Ly=100, B=1, q=3)
# plot_spectrum_edge(Lx=60, Ly=100, B=1./3, bc=1)
plot_spectrum_edge(Lx=60, Ly=100, B=2./5, bc=0)
plt.show()