from __future__ import absolute_import, division, print_function
import numpy as np

"""
This file defines functions that calculate properties from free fermion Hamiltonians
"""

def energies(H):
    """
    Calculate energy spectrum from given Hamiltonian
    input:  H   Hamiltonian
    output:     energy spectrum (in ascending order)
    """
    return np.linalg.eigvalsh(H)


def energy_gap(es):
    """
    Calculate energy gap near E=0 from energy spectrum
    input:  es  energy spectrum
    output:     energy gap
    """
    pos = es[np.where(es>=0)]
    neg = es[np.where(es<0)]
    return np.min(pos) - np.max(neg)


def energy_gap_from_H(H, D=None):
    """
    Calculate energy gap from Hamiltonian
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
    output:         energy gap
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    pos = D[np.where(D>=0)]
    neg = D[np.where(D<0)]
    return np.min(pos) - np.max(neg)


def get_filling(es):
    """
    Calculate filling from energy spectrum
    input:  es  energy spectrum
    output:     filling percentage
    """
    zeros = es[np.argwhere(np.abs(es)<1e-6).ravel()]
    if len(zeros)>0:
        print("get {:d} energies near zero:".format(len(zeros)))
        for z in zeros:
            print("{:.2e}".format(z))
    return sum(es<0) / len(es)


def density_matrix(filled):
    """
    Calculate density matrix from filled states
    input:  filled  filled states with states as columns
    output:         density matrix
    """
    return np.dot(np.conjugate(filled), np.transpose(filled))


def density_matrix_from_H(H, D=None, filling=0.5, keepR=True):
    """
    Calculate density matrix from Hamiltonian
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            filling electron filling, default 0.5, if None fill all negative-energy states
            keepR   indicate whether to keep only non-redundant part, default True.
                    If true, keep only real part of the lower triangular and imaginary part of the upper triangular
    output:         density matrix
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H

    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(len(D)*filling + 1e-10)
    rho = np.dot(np.conjugate(U[:,:filled]), np.transpose(U[:,:filled]))
    if keepR:
        temp = np.ones(rho.shape) - np.triu(1+1j*np.ones(rho.shape), k=1)
        rho = np.real(rho * temp)
    return rho


def entanglement_spectrum(rdm):
    """
    Calculate entanglement spectrum from reduced density matrix
    input:  rdm     reduced density matrix
    output:         entanglement spectrum (in ascending order)
    """
    es = np.linalg.eigvalsh(rdm)
    return es


def entanglement_spectrum_from_H(H, D=None, filling=0.5, cut=0.5):
    """
    Calculate entanglement spectrum from Hamiltonian
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            filling electron filling, default 0.5, if None fill all negative-energy states
            cut     position of the cut, default 0.5
    output:         entanglement spectrum (in ascending order)
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H

    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(len(D)*filling + 1e-10)
    keep = int(len(D)*cut + 1e-10)
    u, s, v = np.linalg.svd(U[:keep, :filled])
    return (s[::-1])**2


def entanglement_gap(es):
    """
    Calculate entanglement gap from entanglement spectrum
    input:  es      entanglement spectrum
    output:         mid gap in entanglement spectrum
    """
    pos = es[np.where(es>=0.5)]
    neg = es[np.where(es<0.5)]
    return np.min(pos) - np.max(neg)


def entanglement_entropy(es):
    """
    Calculate entanglement entropy from entanglement spectrum
    input:  es      entanglement spectrum, one or multi-dimension numpy array
                    entanglement spectrum must in the last dimension
    output:         entanglement entropy
    """
    if es.ndim==1:
        return - np.sum( (1-es)*np.log(1-es+1e-10)+es*np.log(es+1e-10) )
    else:
        return - np.sum( (1-es)*np.log(1-es+1e-10)+es*np.log(es+1e-10), axis=-1 )


def entanglement_entropy_from_H(H, D=None, filling=0.5, cut=(0,0.5)):
    """
    Calculate entanglement spectrum from Hamiltonian
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            filling electron filling, default 0.5, if None fill all negative-energy states
            cut     position of the two cuts, default (0,0.5), i.e. central cut
    output:         entanglement spectrum (in ascending order)
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H
    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(len(D)*filling + 1e-10)
    start = int(len(D)*cut[0] + 1e-10)
    keep = int(len(D)*cut[1] + 1e-10)
    u, s, v = np.linalg.svd(U[start:keep, :filled])
    es = s**2
    return - np.sum( (1-es)*np.log(1-es+1e-10)+es*np.log(es+1e-10) )


def entanglement_entropy_vs_l(H, D=None, filling=0.5, start=1, step=1):
    """
    Calculate entanglement spectrum from Hamiltonian for all possible cuts
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            filling electron filling, default 0.5, if None fill all negative-energy states
            start   smallest numer of sites on the left of the cut, default 1
            step    number of sites between nearest cuts, default 1
    output:         entanglement spectrum (in ascending order)
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H
    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(len(D)*filling + 1e-10)
    ees = []
    for keep in range(start,len(D), step):
        u, s, v = np.linalg.svd(U[:keep, :filled])
        es = s**2
        ees.append( - np.sum( (1-es)*np.log(1-es+1e-10)+es*np.log(es+1e-10) ) )
    return np.array(ees)


def charge_density(H, D=None, filling=0.5):
    """
    Calculate charge density from Hamiltonian
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            filling electron filling, default 0.5, if None fill all negative-energy states
    output:         charge density array
    """
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H
    L = len(D)
    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(L*filling + 1e-10)
    return np.sum(np.abs(U[:,:filled])**2, axis=1)


def topological_index_chiral(rho):
    """
    Calculate real space topological index for chiral symmetric systems defined in Ian's paper PRL.113.046802
    input:  rho     density matrix
    output:         topological invariant
    """
    # position operator
    n = rho.shape[0]//2
    X = np.arange(n)
    q = -rho[::2,1::2]
    qd = -rho[1::2,::2]
    return 4 * np.real(-np.trace(np.dot(qd*X, q)-np.dot(qd, q*X))) / n


def topological_index_chiral_general(H, S=np.array([[0,1],[1,0]])):
    """
    Calculate real space topological index for chiral symmetric systems defined in Ian's paper PRL.113.046802
    input:  H       single particle Hamiltonian
            S       the chiral symmetry operator, default Sigma_x
    output:         topological invariant
    """
    D, U = np.linalg.eigh(S)
    assert all(np.round(D**2,0)==1), "S is not a projector"
    D = D[::-1]
    U = U[:,::-1]
    Sp = ( U * np.where(D>0,1,0) ).dot(U.T.conjugate())
    Sm = ( U * np.where(D<0,1,0) ).dot(U.T.conjugate())
    n = H.shape[0]
    m = S.shape[0]
    L = n//m
    S = np.kron( np.eye(L), S)
    assert np.sum(np.abs(H.dot(S)+S.dot(H)))<0.1*2*n+1e-6, "S is not chiral symmetry SH=-HS"
    Sp = np.kron( np.eye(L), Sp)
    Sm = np.kron( np.eye(L), Sm)
    D, U = np.linalg.eigh(H)
    Q = ( U * np.where(D>0,1,-1) ).dot(U.T.conjugate())
    Qmp = Sm.dot(Q).dot(Sp)
    Qpm = Sp.dot(Q).dot(Sm)
    X = np.diag( np.repeat(range(L), m) )
    return np.real( -np.trace( Qmp.dot( X.dot(Qpm) - Qpm.dot(X) ) ) ) / L


def topological_index_chiral_from_H(H, D=None):
    """
    Calculate real space topological index for chiral symmetric systems defined in Ian's paper PRL.113.046802
    input:  H       single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
    output:         topological invariant
    """
    n = H.shape[0]//2
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H
    # flatband Hamiltonian
    Q = -np.dot(U[:,:n], np.transpose(np.conjugate(U[:,:n])))
    # position operator
    X = np.arange(n)
    q = Q[::2,1::2]
    qd = Q[1::2,::2]
    return 4 * np.real(-np.trace(np.dot(qd*X, q)-np.dot(qd, q*X))) / n


def charge_polarization(density, nums=2):
    """
    Calculate charge polarization from charge density
    input:  density     1D array of charge density
            nums        number of sites in one unit cell, default 2
    output:             charge density, a number in [0,1)
    """
    n = len(density)//nums
    X = np.repeat(np.arange(n), nums)
    # p = np.dot(density-0.5, X)/n
    p = np.dot(density, X)/n - np.mean(X)
    return n+p-int(n+p+1e-6)


def charge_polarization_from_H(H, D=None, nums=2, filling=0.5):
    """
    Calculate charge polarization from Hamiltonian
    input:  H           single particle Hamiltonian or eigen-vectors if D is not None
            D       eigen-values, default None. If give, H is the corresponding eigen-vectors.
            nums        number of sites in one unit cell, default 2
            filling electron filling, default 0.5, if None fill all negative-energy states
    output:             charge density, a number in [0,1)
    """
    L = H.shape[0]
    if D is None:
        D, U = np.linalg.eigh(H)
    else:
        U = H
    if filling is None:
        filled = sum(D<0)
    else:
        filled = int(L*filling + 1e-10)
    density = np.sum(np.abs(U[:,:filled])**2, axis=1)
    n = L//nums
    X = np.repeat(np.arange(n), nums)
    p = np.dot(density-0.5, X)/n
    return n+p-int(n+p+1e-6)


def winding_number_chiral(L=10, t=1, m=0, plotit=True):
    """
    Calculate winding number for chiral symmetric systems defined in Ian's paper PRL.113.046802
    input:  L       system size, default 10
            t       hopping, default 1
            m       mass, default 0
            plotit  indicate whether to plot results, default True
    output:         winding number
    """
    k = np.linspace(0,2*np.pi, L, endpoint=False)
    x = t/(t-1j*m*np.exp(-1j*k))
    if plotit:
        plt.plot(k, x.real, '-')
        plt.plot(k, x.imag, '-')
        plt.show()
    return np.sum(x).real/L


def localization_length(t=1, m=0, w1=0, w2=0):
    """
    Compute localization length of the Chiral model in Ian's paper PRL.113.046802.
    This function returns the value inside the logarithm. The localization length
    diverges when this function is equal to one.
    input:  t,  m, w1, w2   are parameters of the model. They can be number or numpy array.
    returns                 number of numpy array, depending on the input
    """
    tol = 1e-3
    result = t*0 + m*0 + w1*0 + w2*0
    if np.array(result).shape==():
        if np.abs(w1)<tol:
            left = 0.5 / t
        else:
            left = np.power(np.abs(2*t-w1), t/w1-0.5) \
                 / np.power(np.abs(2*t+w1), t/w1+0.5)
        if np.abs(w2)<tol:
            right = 2 * m
        else:
            right = np.power(np.abs(2*m+w2), m/w2+0.5) \
                  / np.power(np.abs(2*m-w2), m/w2-0.5)
    else:
        t = t + result;
        m = m + result;
        w1 = w1 + result;
        w2 = w2 + result;
        left = 0.5 / t
        sel = np.abs(w1)>tol
        left[sel] = np.power(np.abs(2*t[sel]-w1[sel]), t[sel]/w1[sel]-0.5) \
                  / np.power(np.abs(2*t[sel]+w1[sel]), t[sel]/w1[sel]+0.5)
        right = 2 * m
        sel = np.abs(w2)>tol
        right[sel] = np.power(np.abs(2*m[sel]+w2[sel]), m[sel]/w2[sel]+0.5) \
                   / np.power(np.abs(2*m[sel]-w2[sel]), m[sel]/w2[sel]-0.5)
        sel = (w1!=0)&(w2!=0)
    return left*right


if __name__ == '__main__':
    import argparse
    import properties
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="A collection of properties")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given property')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(properties, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(properties, FLAGS.doc).__doc__)
