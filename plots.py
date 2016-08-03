from numpy.linalg import eigh, det, svd
import numpy as np
import matplotlib.pyplot as plt

import functions as func
import operators as op
from operators import TOL
from hamiltonian import *

def plot_energy_split(func, k, **arg):
	''' Plot the energy split
		-> func: the function to calculate the Hamiltonian
		-> k: the number of energy levels to be plotted
		-> the parameters of the Hamiltonian '''

	print "\n********** Plot the energy splitting as a function of system size **********"

	H = func(2, **arg)
	sym = H.sym

	min_l = int( np.log(k*sym)/np.log(sym) ) + 1
	max_l = int( np.log(1e4*sym)/np.log(sym) )
	val_l = np.arange(min_l, max_l)
	ene = np.zeros((val_l.shape[0], k, sym))

	for i,L in enumerate(val_l):
		H = func(L, **arg)
		for j in range(sym):
			# cc = fermion_construct(L, 2, fermion_c_d(), 0, fermion_c(), L-1)
			print "\t calculate eigenvalues for symmetry sector %d ..." % j
			D, U = eigh(H.val[j][j])
			ene[i,:,j] = D[:k]
			# print np.diag( np.dot(U[:,:1].T, np.dot(cc.val[0][0], U[:,:1])) )
			# D, U = eigh(H.val[1][1])
			# ene1[i,:] = D[:k]
			# print np.diag( np.dot(U[:,:1].T, np.dot(cc.val[0][0], U[:,:1])) )
	plt.figure("Energy split")
	name = ""
	for kw in arg.keys():
		if arg[kw]!=0:
			name += kw + "=" + str(arg[kw]) + "  "
	plt.suptitle(name)
	for j in range(sym-1):
		plt.subplot(2, sym-1, j+1)
		for i in range(k):
			diff = ene[:,i,j+1] - ene[:,i,0]
			plt.plot(val_l+0.2*i, diff, 'o')
		plt.xticks(val_l)
		plt.subplot(2, sym-1, j+1+sym-1)
		for i in range(k):
			diff = ene[:,i,j+1] - ene[:,i,0]
			plt.plot(val_l+0.2*i, np.abs(diff), 'o')
		plt.xticks(val_l)
		plt.yscale('log')


def plot_entanglement_entropy(func, fit=False, level=0, **arg):
	''' Plot the entanglement entropy as a function of subsystem size
		-> func: the function to calculate the Hamiltonian
		-> level: the energy level to plot
		-> the parameters of the Hamiltonian '''

	print "\n********** Plot the entanglement entropy at each cut **********"

	H = func(**arg)
	L = H.L
	sym = H.sym

	plt.figure("Entanglement entropy")
	name = ""
	for kw in arg.keys():
		if arg[kw]!=0:
			name += kw + "=" + str(arg[kw]) + "  "
	plt.title(name)

	ee = np.zeros(L-1)
	for s in range(sym):
		print "\t calculate eigenvalues for symmetry sector %d ..." % s
		D, U = eigh(H.val[s][s])
		for l in range(1,L):
			ee[l-1] = entanglement_entropy(U[:,level], l, H, s=s)
		plt.plot(np.arange(1,L), ee, 'o', label="sector %d" % s)
		if fit:
			p = fit_central_charge(ee)
			x = np.linspace(1,L-1,100,endpoint=True)
			plt.plot(x, np.polyval(p, np.log(np.sin(np.pi*x/L))/6), '-', label="c=%.2f"%p[0])
	plt.xticks(range(L+1))
	plt.legend(loc='best')


def plot_energy_distribution(func, **arg):
	''' Plot the energy distribution
		-> func: the function to calculate the Hamiltonian
		-> the parameters of the Hamiltonian '''

	print "\n********** Plot the energy distribution **********"

	H = func(**arg)
	sym = H.sym

	plt.figure("energy distribution")
	name = ""
	for kw in arg.keys():
		if arg[kw]!=0:
			name += kw + "=" + str(arg[kw]) + "  "
	plt.title(name)

	evals = []
	for s in range(sym):
		print "\t calculate eigenvalues for symmetry sector %d ..." % s
		D, U = eigh(H.val[s][s])
		evals.append(D)

	for s in range(len(evals))[::-1]:
		plt.subplot(121)
		plt.plot(evals[s], 'o', label="sector %d" % s)
		plt.subplot(122)
		plt.hist(np.array(evals[:s+1]).ravel(), 75)
	plt.subplot(121)
	plt.legend(loc='best')
	plt.xticks([])


def plot_SSH_green_wk(dt=1.):
	''' plot the greens function in the k-w plane '''
	L = 12
	num = 51	# number of energies to calculate
	ws = np.linspace(-3,3,num)
	H = SSH(L, t=1., d=dt, pbc=True)
	evals, evecs = func.solve_hamiltonian(H)

	f, ax = plt.subplots(2,2)#, sharex='col', sharey='row')

	d_num = L/2+1
	Gij = func.green(ws, op.fermion_c(), L, evals, evecs)
	lim = 2
	cdict = {'red':   [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0,  1.0, 1.0)],
			'green': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0,  0.0, 0.0)],
			'blue':  [(0.0, 1.0, 1.0), (0.5, 1.0, 1.0), (1.0,  0.0, 0.0)]}
	from matplotlib.colors import LinearSegmentedColormap
	cmap = LinearSegmentedColormap('mycm', cdict)

	for x1 in range(2):
		for x2 in range(2):
			Gd = np.zeros((num, d_num), dtype=np.complex)
			for i in range(num):
				Gd[i, :] =  func.fourier(Gij[i][x1,x2::2])
			# plt.figure("%d-%d"%(x1,x2))
			im = ax[x1][x2].imshow(np.real(Gd), interpolation='nearest', origin='lower',
									extent=(0,np.pi*2,-3.,3.), aspect='auto', cmap=cmap)
			im.set_clim(-lim,lim)
			# ax[x1][x2].set_xticks([0, np.pi, np.pi*2], [-1,0,1])
			# ax[x1][x2].set_yticks([-3.,0,3.], [-3,0,3])
			# im.colorbar(ticks=[-lim,0,lim], label="G(w,k)")


def plot_Hubbard_green_wk():
	L=6
	H = Hubbard(L, t=1., u=1., f=1., pbc=True)
	evals, evecs = func.solve_hamiltonian(H)

	sec= 0
	ge = evals[0][0]
	for s in range(evals.shape[0]):
		if evals[s][0] < ge:
			ge = evals[s][0]
			sec = s
	num = op.fermion_up().hermitian().mult( op.fermion_up() )
	num.plus( op.fermion_down().hermitian().mult( op.fermion_down() ) )
	density = func.expectation(L, evecs[sec][:,0], sec, [num] )

	num = 11
	ws = np.linspace(-3,3,num)
	Gij = func.green(ws, op.fermion_up(), L, evals, evecs)
	Gwk = np.zeros((num, L+1), dtype=np.complex)
	for i in range(num):
		Gwk[i, :] =  func.fourier(Gij[i][0,:])
	print Gij[(num-1)/2][0,:]
	print func.fourier(Gij[(num-1)/2][0,:])
	plt.figure("Green")
	plt.imshow(np.imag(Gwk), interpolation='nearest', aspect='auto')
	plt.colorbar()
	print "density of the state:", density


def plot_SSH_green():
	''' plot the Det(G) in the plane dt-w'''
	L = 8
	num = 51
	d_num = 20
	ws = np.linspace(-3,3,num)
	Gd = np.zeros((num, d_num), dtype=np.complex)
	for j, d in enumerate(np.linspace(-1,1,d_num)):
		H = SSH(L, t=1., d=d, pbc=False)
		evals, evecs = func.solve_hamiltonian(H)

		Gwk = np.zeros(num, dtype=np.complex)
		Gij = func.green(ws, op.fermion_c(), L, evals, evecs)
		for i in range(num):
			Gwk[i] =  det(Gij[i])
		Gd[:,j] = Gwk

		# plt.figure("Green u=%.2f"%d)
		# plt.plot(ws, np.real(Gwk),'o-')
		# plt.ylim([-1,1])
		# print np.min( np.imag(Gwk) ), np.max( np.imag(Gwk) )

	plt.figure("poles")
	fig = plt.imshow(np.real(Gd), interpolation='nearest', aspect=0.4)
	fig.set_clim(-1,1)
	plt.colorbar(ticks=[-1,0,1],label="Det(G)")
	plt.xticks([0.5,d_num/2.,d_num-0.5], [-1,0,1], fontsize=20)
	plt.yticks([0.5,num/2.,num-0.5], [3,0,-3], fontsize=20)
	plt.xlabel('d', fontsize=20)
	plt.ylabel('w', fontsize=20)


def plot_Kitaev_green():
	''' plot the Det(G) in the plane u-w'''
	L = 8
	num = 51
	u_num = 20
	ws = np.linspace(-3,3,num)
	Gd = np.zeros((num, u_num), dtype=np.complex)
	for j, u in enumerate(np.linspace(0,2,u_num)):
		H = Kitaev(L, t=1., u=u, d=1., pbc=False)
		evals, evecs = func.solve_hamiltonian(H)

		Gwk = np.zeros(num, dtype=np.complex)
		Gij = func.green(ws, op.fermion_c(), L, evals, evecs)
		for i in range(num):
			Gwk[i] =  det(Gij[i])
		Gd[:,j] = Gwk

	plt.figure("poles")
	fig = plt.imshow(np.real(Gd), interpolation='nearest', aspect=0.4)
	fig.set_clim(-1,1)
	plt.colorbar(ticks=[-1,0,1],label="Det(G)")
	plt.xticks([0.5,u_num/2.,u_num-0.5], [0,1,2], fontsize=20)
	plt.yticks([0.5,num/2.,num-0.5], [3,0,-3], fontsize=20)
	plt.xlabel('u', fontsize=20)
	plt.ylabel('w', fontsize=20)


def plot_Peierls_Hubbard_green(u=0.0):
	''' plot the Det(G) in the plane dt-w'''
	L = 6
	num = 51
	d_num = 20
	ws = np.linspace(-3,3,num)
	Gd = np.zeros((num, d_num), dtype=np.complex)
	for j, d in enumerate(np.linspace(-1,1,d_num)):
		H = Peierls_Hubbard(L, t=1., dt=d, U=u, pbc=False)
		evals, evecs = func.solve_hamiltonian(H)

		Gwk = np.zeros(num, dtype=np.complex)
		Gij_up = func.green(ws, op.fermion_up(), L, evals, evecs)
		Gij_down = func.green(ws, op.fermion_down(), L, evals, evecs)
		for i in range(num):
			Gwk[i] =  det(Gij_up[i])*det(Gij_down[i])
		Gd[:,j] = Gwk

		# plt.figure("Green u=%.2f"%d)
		# plt.plot(ws, np.real(Gwk),'o-')
		# plt.ylim([-1,1])
		# print np.min( np.imag(Gwk) ), np.max( np.imag(Gwk) )

	plt.figure("poles")
	fig = plt.imshow(np.real(Gd), interpolation='nearest', aspect=0.4)
	fig.set_clim(-1,1)
	plt.colorbar(ticks=[-1,0,1],label="Det(G)")
	plt.xticks([0.5,d_num/2.,d_num-0.5], [-1,0,1], fontsize=20)
	plt.yticks([0.5,num/2.,num-0.5], [3,0,-3], fontsize=20)
	plt.xlabel('d', fontsize=20)
	plt.ylabel('w', fontsize=20)


def calc_disconnected_EE():
	L = 10
	H = Kitaev(L, t=1., d=1., u=0., pbc=False)
	# H = SSH(L, t=1., d=0., pbc=False)
	# H = AKLT(L, pbc=False)
	evals, evecs = func.solve_hamiltonian(H)
	sec = np.argmin(evals[:,0])
	print "\nEnergies (ground sec=%d):\n" % sec, evals[:,:5]
	base = basis(L, H.basis, H.sym_sum)
	dim = H.basis.ravel().shape[0]
	print ""

	rho = func.discrete_reduced_density_matrix(evecs[sec,:,0], base[sec], [[0,L]], dim)
	u, d, v = svd(rho)
	ee = -np.sum(d**2 * np.log( d**2+TOL))
	print "Entanglement entropy for full:", ee, "or %.2f ln2" % (ee/np.log(2))

	rho = func.discrete_reduced_density_matrix(evecs[sec,:,0], base[sec], [[0,(L/4)*2]], dim)
	u, d, v = svd(rho)
	ee = -np.sum(d**2 * np.log( d**2+TOL))
	print "Entanglement entropy for half:", ee, "or %.2f ln2" % (ee/np.log(2))

	rho = func.discrete_reduced_density_matrix(evecs[sec,:,0], base[sec], [[2,6]], dim)
	u, d, v = svd(rho)
	ee = -np.sum(d**2 * np.log( d**2+TOL))
	print "Entanglement entropy for central:", ee, "or %.2f ln2" % (ee/np.log(2))

	rho = func.discrete_reduced_density_matrix(evecs[sec,:,0], base[sec], [[0,2],[6,8]], dim)
	u, d, v = svd(rho)
	ee = -np.sum(d**2 * np.log( d**2+TOL))
	print "Entanglement entropy for edge and central:", ee, "or %.2f ln2" % (ee/np.log(2))

	rho = func.discrete_reduced_density_matrix(evecs[sec,:,0], base[sec], [[2,4],[6,8]], dim)
	u, d, v = svd(rho)
	ee = -np.sum(d**2 * np.log( d**2+TOL))
	print "Entanglement entropy for two parts:", ee, "or %.2f ln2" % (ee/np.log(2))


def calc_disconnected_EE_free(L=100):
	H = Kitaev_free(L, t=1., d=1., u=0., pbc=False)
	# H = SSH_free(L=L, t=1., d=0., pbc=False)
	D, U = eigh(H)
	size = H.shape[0]
	cc = np.dot(U[:,:size/2], U[:,:size/2].T)

	ee = np.zeros(size/2-1)
	for i, q in enumerate(range(2, size, 2)):
		evals, evecs = eigh( cc[:q,:q] )
		ee[i] = -np.sum( evals * np.log(evals+1e-10) )
		evals, evecs = eigh( cc[q:,q:] )
		ee[i] += -np.sum( evals * np.log(evals+1e-10) )
	plt.plot(np.arange(1,size/2), ee/np.log(2), 'o')
	plt.ylim([-0.1,1.6])
	plt.xticks([0,size/2])
	print np.polyfit(np.log(np.sin(np.pi*np.arange(1,size/2)/(size/2)))/6., ee/2, 1)
