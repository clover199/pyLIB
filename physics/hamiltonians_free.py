from __future__ import absolute_import, division, print_function
import numpy as np

from numpy.linalg import eigh,eig
from random import random

"""
This file defines free fermion Hamiltonians
"""

def quadrupole(L=10, lambda_x=1, lambda_y=1, gamma_x=1, gamma_y=1, delta=0, QP=-1):
    """
    Quadrupole Hamiltonian in EQ 6.28 of the PRB paper.

    L: number of sites, default 10
    lambda_x: number or L*L-array, default 1
    lambda_y: number or L*L-array, default 1
    gamma_x: number or L*L-array, default 1
    gamma_y: number or L*L-array, default 1
    delta: breaks degeneracy
    QP: set to -1 for "real QP model", 1 for "fake QP model", default -1

        # basis: x,y,n
        (x,y,0) lambda_x (x+1,y,2), (x,y,0) lambda_y (x,y+1,3)
        (x,y,3) lambda_x (x+1,y,1), (x,y,2) lambda_y (x,y+1,1) * QP
        (x,y,0) gamma_x (x,y,2),    (x,y,0) gamma_y (x,y,3)
        (x,y,1) gamma_x (x,y,3),    (x,y,1) gamma_y (x,y,2) * QP
        (x,y,0)  delta (x,y,0)
        (x,y,1)  delta (x,y,1)
        (x,y,2) -delta (x,y,2)
        (x,y,3) -delta (x,y,3)

    function returns a (4L,4L) numpy array
    """

    n = L*L*4
    H = np.zeros([n,n])
    i = np.arange(0,n,4)
    ix = i+4
    ix[L-1::L] -= 4*L
    iy = i+L*4
    iy[-L:] -= L*4*L

    H[i, ix+2] = lambda_x
    H[i+3, ix+1] = lambda_x
    H[i, iy+3] = lambda_y
    H[i+2, iy+1] = lambda_y * QP
    H[i, i+2] = gamma_x
    H[i+1, i+3] = gamma_x
    H[i, i+3] = gamma_y
    H[i+1, i+2] = gamma_y * QP
    H[i, i] = delta/2.0
    H[i+1, i+1] = delta/2.0
    H[i+2, i+2] = -delta/2.0
    H[i+3, i+3] = -delta/2.0

    return H + H.T.conjugate()


def AIII(L=10, t=1, m=0, s=0):
    """
    Chiral chain

    H = 1/2 t_j c^d_j ( sigma_x + i sigma_y ) c_j+1 + h.c.
        + m_j c^d_j sigma_y c_j
        + s c^d_j sigma_z c_j

    L: number of sites, default 10
    t: number or L-array of hopping values, default 1
    m: number or L-array of mass values, default 0
    s: symmetry breaking strength

        # basis: 1A    1B    2A    2B ...
        #  1A     s  -i m          t
        #  1B   i m    -s
        #  2A                 s  -i m
        #  2B    t          i m    -s

    function returns a (2L,2L) numpy array
    """

    H = np.zeros((2*L,2*L), dtype=complex)
    i = np.arange(0,2*L,2)
    H[i, (i+3)%(2*L)] = t
    H[i, i+1] = -1j*m
    H[i, i] = s/2.0
    H[i+1, i+1] = -s/2.0

    return H + H.T.conjugate()


def SSH(L=10, t=1, m=0, s=0):
    """
    Su-Schrieffer-Heeger chain

    H = - t_j c^d_Bj c_Aj+1 + h.c.
        - m_j c^d_Aj c_Bj + h.c.
        + s c^d_Aj c_Aj - s c^d_Bj c_Bj

    L: number of sites, default 10
    t: number or L-array of inter-cell hopping values, default 1
    m: number or L-array of intra-cell hopping values, default 0
    s: symmetry breaking strength

        # basis: 1A   1B   2A   2B ...
        #  1A     s   -m
        #  1B    -m   -s   -t
        #  2A         -t    s   -m
        #  2B              -m   -s

    function returns a (2L,2L) numpy array
    """

    H = np.zeros((2*L,2*L), dtype=float)
    i = np.arange(0,2*L,2)
    H[i+1, (i+2)%(2*L)] = -t
    H[i, i+1] = -m
    H[i, i] = s/2.0
    H[i+1, i+1] = -s/2.0

    return H + H.transpose()


def pSC(L=10, t=1, d=1, h=0, s=0):
    """
    p-wave superconductor

    H = - t_j c^d_j c_j+1 + h.c.
        + d_j c_j c_j+1 + h.c.
        + 2 h_j c^d_j c_j

    L: number of sites, default 10
    t: number or L-array of hoppting values, default 1
    d: number or L-array of pairing values, default 1
    h: number or L-array of mass values, default 0
    s: symmetry breaking strength

        # basis: 1c   1d   2c   2d ...
        #  1d     h    s  -t/2 -d/2
        #  1c     s   -h   d/2  t/2
        #  2d   -t/2  d/2   h    s
        #  2c   -d/2  t/2   s   -h

    function returns a (2L,2L) numpy array
    """

    H = np.zeros((2*L,2*L))
    i = np.arange(0,2*L,2)
    H[i, (i+2)%(2*L)] = -t/2
    H[i+1, (i+3)%(2*L)] = t/2
    H[i, (i+3)%(2*L)] = -d/2
    H[i+1, (i+2)%(2*L)] = d/2
    H[i, i] = h/2
    H[i+1, i+1] = -h/2
    H[i, i+1] = s

    return H + H.T.conjugate()


def Kitaev(L=10, u=0, v=1, s=0):
    """
    Kitaev chain

    H = i u_j a_j b_j + i v_j b_j a_j+1
    p-wave superconductor with t=d=v, h=u

    L: number of sites, default 10
    u: number or L-array of mass values, default 0
    v: number or L-array of hoppting values, default 1
    s: symmetry breaking strength, default 0

    function returns pSC(L, v, v, u)
    """

    return pSC(L, v, v, u, s)


def double_Kitaev(L=10, t1=1, h1=0, t2=1, h2=0, s=0):
    """
    double Kitaev chain with term breaking time-reversal symmetry
        i s_j a_1j a_2j - i s_j b_1j b_2j

    # basis: 1c   1d   1c   1d   2c   2d ...
    #  1d    h1            i s  -t/2 -t/2
    #  1c        -h1  i s       t/2  t/2
    #  1d        -i s  h2
    #  1c   -i s         -h2

    L: number of sites, default 10
    t1: number or L-array of hoppting values for chain 1, default 1
    h1: number or L-array of mass values for chain 1, default 0
    t2: number or L-array of hoppting values for chain 2, default 1
    h2: number or L-array of mass values for chain 2, default 0
    s: number or L-array of symmetry breaking values, default 0

    function returns a (4L,4L) numpy array
    """

    H = np.zeros((4*L,4*L), dtype=complex)
    i = np.arange(0,4*L,4)
    H[i, (i+4)%(4*L)] = -t1/2.0
    H[i, (i+5)%(4*L)] = -t1/2.0
    H[i+1, (i+4)%(4*L)] = t1/2.0
    H[i+1, (i+5)%(4*L)] = t1/2.0
    H[i+2, (i+6)%(4*L)] = -t2/2.0
    H[i+2, (i+7)%(4*L)] = -t2/2.0
    H[i+3, (i+6)%(4*L)] = t2/2.0
    H[i+3, (i+7)%(4*L)] = t2/2.0
    H[i, i] = h1/2.0
    H[i+1, i+1] = -h1/2.0
    H[i+2, i+2] = h2/2.0
    H[i+3, i+3] = -h2/2.0
    H[i+1, i+2] = 1j*s;
    H[i, i+3] = 1j*s;

    return H + H.transpose().conjugate()


def quick_set(L, val, boundary=1, disorder=0):
    """
    used to set parameter values quickly

    L: number of sites
    val: number
    boundary: 0 or 1 for boundary condition OBC or PBC
    disorder: number, width of the uniform distribution

    function returns a (L,) numpy array
    """

    vals = val + disorder*(np.random.random(L)-0.5)
    vals[-1] = vals[-1]*boundary
    return vals


def quick_set_x(L, val, boundary=1, disorder=0):
    """
    used to set parameter values for x-direction quickly for 2D systems

    L: number of sites in one direction
    val: number
    boundary: 0 or 1 for boundary condition OBC or PBC
    disorder: number, width of the uniform distribution

    function returns a (L*L,) numpy array
    """

    vals = val + disorder*(np.random.random([L,L])-0.5)
    vals[:,-1] = vals[:,-1]*boundary
    return vals.ravel()


def quick_set_y(L, val, boundary=1, disorder=0):
    """
    used to set parameter values for y-direction quickly for 2D systems

    L: number of sites in one direction
    val: number
    boundary: 0 or 1 for boundary condition OBC or PBC
    disorder: number, width of the uniform distribution

    function returns a (L*L,) numpy array
    """

    vals = val + disorder*(np.random.random([L,L])-0.5)
    vals[-1,:] = vals[-1,:]*boundary
    return vals.ravel()


def disorder_quadrupole(L=10, lambda_x=1, lambda_y=1, gamma_x=1, gamma_y=1, Wlambda=0, Wgamma=0, delta=0, boundary=0, QP=-1):
    """
    Disordered quadrupole Hamiltonian in EQ 6.28 of the PRB paper.

    L: number of sites, default 10
    lambda_x: default 1
    lambda_y: default 1
    gamma_x: default 1
    gamma_y: default 1
    Wlambda: parameterize the disorder along the lambda-bonds
    Wgamma: parameterize the disorder along the gamma-bonds
    delta: breaks degeneracy
    boundary: 0 or 1 for boundary condition OBC or PBC
    QP: set to -1 for "real QP model", 1 for "fake QP model", default -1

    function returns a (4L,4L) numpy array
    """

    return quadrupole(L=L,
                      lambda_x=quick_set_x(L, lambda_x, boundary, Wlambda),
                      lambda_y=quick_set_y(L, lambda_y, boundary, Wlambda),
                      gamma_x=quick_set_x(L, gamma_x, 1, Wgamma),
                      gamma_y=quick_set_y(L, gamma_y, 1, Wgamma),
                      delta=delta,
                      QP=QP)


def disorderAIII(L=10, t=1, m=1, W1=0, W2=0, s=0, boundary=0):
    """
    Disordered chiral chain. Hamiltonian in Ian's paper PRL.113.046802

    L: number of sites, default 10
    t: number, hopping term, default 1
    m: number, mass term, default 1
    W1: number, disorder for hopping t, default 0
    W2: number, disorder for mass m, default 0
    s: symmetry breaking strength
    boundary: 0 or 1 for boundary condition OBC or PBC

    function returns a (2L,2L) numpy array
    """

    return AIII(L=L,
                t=quick_set(L,t,boundary,W1),
                m=quick_set(L,m,1,W2),
                s=s)


def easySSH(L=10, t=1, m=1, s=0, boundary=0):
    """
    A quick way to get SSH chain Hamiltonian

    L: number of sites, default 10
    t: number of hoppting values, default 1
    m: number, mass term, default 1
    s: symmetry breaking strength, default 0
    boundary: 0 or 1 for boundary condition OBC or PBC, default 0

    function returns a (2L,2L) numpy array
    """
    return SSH(L=L,
               t=quick_set(L,t,boundary,0),
               m=m,
               s=s)


def disorderSSH(L=10, t=1, m=1, W1=0, W2=0, s=0, boundary=0):
    """
    Disordered SSH chain

    L: number of sites, default 10
    t: number, hopping term, default 1
    m: number, mass term, default 1
    W1: number, disorder for hopping t, default 0
    W2: number, disorder for mass m, default 0
    s: symmetry breaking strength
    boundary: 0 or 1 for boundary condition OBC or PBC

    function returns a (2L,2L) numpy array
    """

    return SSH(L=L,
               t=quick_set(L,t,boundary,W1),
               m=quick_set(L,m,1,W2),
               s=s)


def easy_pSC(L=10, t=1, d=1, h=0, boundary=0):
    """
    A quick way to get the p-wave superconductor Hamiltonian

    L: number of sites, default 10
    t: number of hoppting values, default 1
    d: number of pairing values, default 1
    h: number of mass values, default 0
    boundary: 0 or 1 for boundary condition OBC or PBC, default 0

    function returns a (2L,2L) numpy array
    """
    return pSC(L=L,
               t=quick_set(L,t,boundary,0),
               d=quick_set(L,d,boundary,0),
               h=h)


def disorderKitaev(L=10, t=1, m=1, W1=0, W2=0, s=0, boundary=0):
    """
    Disordered Kitaev chain

    L: number of sites, default 10
    t: number, hopping term, default 1
    m: number, mass term, default 1
    W1: number, disorder for hopping t, default 0
    W2: number, disorder for mass m, default 0
    s: symmetry breaking strength
    boundary: 0 or 1 for boundary condition OBC or PBC

    function returns a (2L,2L) numpy array
    """
    return Kitaev(L=L,
                  u=quick_set(L,m,1,W2),
                  v=quick_set(L,t,boundary,W1),
                  s=s)


if __name__ == '__main__':
    import argparse
    import hamiltonians
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="A collection of free fermion Hamiltonians")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given system')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(hamiltonians, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(hamiltonians, FLAGS.doc).__doc__)
