from __future__ import absolute_import, division, print_function
import numpy as np

"""
This file defines full Hamiltonians from their matrix product operators
"""

def pSC_MPO(t=1.0, d=1.0, h=1.0):
    """
    p-wave superconductor

    H = - t_j c^d_j c_j+1 + h.c.
        + d_j c_j c_j+1 + h.c.
        + 2 h_j c^d_j c_j

    t: hoppting value, default 1
    d: pairing value, default 1
    h: mass value, default 1

    #      I        0          0         0
    #      c        0          0         0
    #     c^d       0          0         0
    # 2h*(n-1/2) -t*c^d+d*c -d*c^d+t*c   I

    function returns a (4,2,2,4) numpy array
    """

    H = np.zeros([4,2,2,4])
    # I
    H[0,0,0,0] = 1
    H[0,1,1,0] = 1
    H[3,0,0,3] = 1
    H[3,1,1,3] = 1
    # n
    H[3,0,0,0] = -h
    H[3,1,1,0] = h
    # c
    H[1,0,1,0] = 1
    H[3,0,1,1] = d
    H[3,0,1,2] = t
    # c^d
    H[2,1,0,0] = 1
    H[3,1,0,1] = -t
    H[3,1,0,2] = -d

    return H


def Kitaev_MPO(u=1.0, v=1.0):
    """
    Kitaev chain

    H = i u_j a_j b_j + i v_j b_j a_j+1
    p-wave superconductor with t=d=v, h=u

    u: mass value, default 1
    v: hoppting value, default 1

    function returns a (4,2,2,4) numpy array
    """

    return pSC(v, v, u)


def Get_Full_Hamiltonian(L, H):
    """
    Construct the full Hamiltonian from the MPO with open boundary conditions

    L: number of sites
    H: the MPO operator for the system

    function returns a (n,n) numpy array with n=d**L, d=H.shape[1]
    """
    result = H[-1,:,:,:]
    for _ in range(1,L-1):
        result = np.tensordot(result, H, axes=[-1,0])
    result = np.tensordot( result, H[:,:,:,0], axes=[-1,0])
    result = np.moveaxis(result,
                         np.concatenate([np.arange(0,2*L,2, dtype=int),
                                         np.arange(1,2*L,2, dtype=int)]),
                         np.arange(2*L, dtype=int))
    d = result.shape[0]
    n = d**L
    return result.reshape([n,n])


if __name__ == '__main__':
    import argparse
    import hamiltonians
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="A collection of Hamiltonians")
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
