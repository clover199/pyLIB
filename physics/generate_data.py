from __future__ import absolute_import, division, print_function
import numpy as np
from time import time

"""
This file defines functions that generates data for different physics problems
"""

from functions import best_time


def generate_data(Ham, pro, names, tmin=0, tmax=10, num=100, repeat=1, filename='data', **kwargs):
    """
    A general function to generate and save data for systems with one changing parameter t
    input:  Ham     Hamiltonian function
            pro     property function that only takes Hamiltonian matrix as input
            names   a dictionary indicating parameter name and relationship with changing t
                    e.g. {"x":1, "y":2} means x=t, y=2t.
            tmin    minimum value of t
            tmax    maximum value of t
            num     number of t values to take
            repeat  number of repeats at one t value
            filename the filename to save the data, default 'data'
                    If None, return data.
            kwargs  other key words to pass to the Hamiltonain function Ham
    output:         data, ts (only when filename=None)
    """
    testH = Ham(**kwargs)
    testp = np.array(pro(testH)).ravel()
    data = np.zeros([num*repeat, testp.shape[0]], testp.dtype)

    print("Generate data:")
    ts = np.repeat(np.linspace(tmin, tmax, num, endpoint=True), repeat)
    begin = time()
    loc = 0
    keys = names.keys()
    mult = np.array(list(names.values()))
    print("-"*100 + " {:.2f} min".format((time()-begin)/60), end='\r' )
    for i, t in enumerate(ts):
        change = dict(zip(keys, mult*t))
        try:
            H = Ham(**kwargs, **change)
            data[i,:] = pro(H).ravel()
        except np.linalg.linalg.LinAlgError:
            H = Ham(**kwargs, **change)
            data[i,:] = pro(H).ravel()
        x = int((i+1)*100/num/repeat)
        if x>loc:
            print("#"*(x) + "-"*(100-x) + " {:.2f} min".format((time()-begin)/60), end='\r')
            loc = x
    print("\nTotal time used:", best_time(time()-begin))
    if filename is None:
        return data, ts
    np.savez(filename, data=data, ts=ts)
    print("Save data to {:s}.npz".format(filename))


def generate_data2(Ham, pro1, pro2, names, tmin=0, tmax=10, num=100, repeat=1, filename='data', **kwargs):
    """
    A general function to generate and save data for systems with one changing parameter t
    Two properties are calculated with one as features and the other as indicators
    input:  Ham     Hamiltonian function
            pro1    property function that only takes Hamiltonian matrix as input
            pro2    property function that only takes Hamiltonian matrix as input
            names   a dictionary indicating parameter name and relationship with changing t
                    e.g. {"x":1, "y":2} means x=t, y=2t.
            tmin    minimum value of t
            tmax    maximum value of t
            num     number of t values to take
            repeat  number of repeats at one t value
            filename the filename to save the data, default 'data'
                    If None, return data.
            kwargs  other key words to pass to the Hamiltonain function Ham
    output:         data, data2, ts (only when filename=None)
    """
    testH = Ham(**kwargs)
    testp = np.array(pro1(testH)).ravel()
    data1 = np.zeros([num*repeat, testp.shape[0]], testp.dtype)
    testp = np.array(pro2(testH)).ravel()
    data2 = np.zeros([num*repeat, testp.shape[0]], testp.dtype)

    print("Generate data:")
    ts = np.repeat(np.linspace(tmin, tmax, num, endpoint=True), repeat)
    begin = time()
    loc = 0
    keys = names.keys()
    mult = np.array(list(names.values()))
    print("-"*100 + " {:.2f} min".format((time()-begin)/60), end='\r' )
    for i, t in enumerate(ts):
        change = dict(zip(keys, mult*t))
        try:
            H = Ham(**kwargs, **change)
            D, U = np.linalg.eigh(H)
            data1[i,:] = pro1(U, D=D).ravel()
            data2[i,:] = pro2(U, D=D).ravel()
        except np.linalg.linalg.LinAlgError:
            H = Ham(**kwargs, **change)
            D, U = np.linalg.eigh(H)
            data1[i,:] = pro1(U, D=D).ravel()
            data2[i,:] = pro2(U, D=D).ravel()
        x = int((i+1)*100/num/repeat)
        if x>loc:
            print("#"*(x) + "-"*(100-x) + " {:.2f} min".format((time()-begin)/60), end='\r')
            loc = x
    print("\nTotal time used:", best_time(time()-begin))
    if filename is None:
        return data1, data2, ts
    np.savez(filename, data=data1, data2=data2, ts=ts)
    print("Save data to {:s}.npz".format(filename))


def generate_2D_data(Ham, pro, names,
                     tmin1=0, tmax1=10, num1=100,
                     tmin2=0, tmax2=10, num2=100,
                     filename='data', **kwargs):
    """
    A general function to generate and save data for systems with
    two changing parameters t1, t2. One property is calculated as features.
    input:  Ham     Hamiltonian function
            pro     property function that only takes Hamiltonian matrix as input
            names   a list indicating names of parameter t1 and t2
            tmin1   minimum value of t1
            tmax1   maximum value of t1
            num1    number of t1 values to take
            tmin2   minimum value of t2
            tmax2   maximum value of t2
            num2    number of t2 values to take
            filename the filename to save the data, default 'data'
                    If None, return data.
            kwargs  other key words to pass to the Hamiltonain function Ham
    output:         data, data2, ts (only when filename=None)
    """
    testH = Ham(**kwargs)
    testp = np.array(pro(testH)).ravel()
    data = np.zeros([num1*num2, testp.shape[0]], testp.dtype)

    print("Generate data:")
    t1s, t2s = np.meshgrid(
        np.linspace(tmin1, tmax1, num1, endpoint=True),
        np.linspace(tmin2, tmax2, num2, endpoint=True) )
    ts = np.vstack([t1s.ravel(),t2s.ravel()]).T
    begin = time()
    loc = 0
    print("-"*100 + " {:.2f} min".format((time()-begin)/60), end='\r' )
    for i, t in enumerate(ts):
        change = dict(zip(names, t))
        try:
            H = Ham(**kwargs, **change)
            data[i,:] = pro(H).ravel()
        except np.linalg.linalg.LinAlgError:
            H = Ham(**kwargs, **change)
            data[i,:] = pro(H).ravel()
        x = int((i+1)*100/ts.shape[0])
        if x>loc:
            print("#"*(x) + "-"*(100-x) + " {:.2f} min".format((time()-begin)/60), end='\r')
            loc = x
    print("\nTotal time used:", best_time(time()-begin))
    if filename is None:
        return data, ts
    np.savez(filename, data=data, ts=ts)
    print("Save data to {:s}.npz".format(filename))


if __name__=="__main__":
    import argparse
    import generate_data
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="Functions to generate data")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given function')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(generate_data, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(generate_data, FLAGS.doc).__doc__)
