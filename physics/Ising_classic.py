import numpy as np
import matplotlib.pyplot as plt
from time import time

"""
This file defines functions used to calculate classic Ising model and generate data for machine learning
"""

# H = - ferro \sum_<i,j> S_i S_j

def state_energy(state, boundary_x, boundary_y):
    energy = -np.sum(state[1:,:]*state[:-1,:])-np.sum(state[:,1:]*state[:,:-1])
    energy += -boundary_x*np.sum(state[:,0]*state[:,-1])
    energy += -boundary_y*np.sum(state[0,:]*state[-1,:])
    return energy


def update(state, energy, temperature, boundary_x, boundary_y, ferro):
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            site_energy = 0
            if i==0:
                site_energy += -ferro*boundary_y*state[i,j]*state[-1,j]
            else:
                site_energy += -ferro*state[i,j]*state[i-1,j]
            if i==state.shape[0]-1:
                site_energy += -ferro*boundary_y*state[i,j]*state[0,j]
            else:
                site_energy += -ferro*state[i,j]*state[i+1,j]
            if j==0:
                site_energy += -ferro*boundary_x*state[i,j]*state[i,-1]
            else:
                site_energy += -ferro*state[i,j]*state[i,j-1]
            if j==state.shape[1]-1:
                site_energy += -ferro*boundary_x*state[i,j]*state[i,0]
            else:
                site_energy += -ferro*state[i,j]*state[i,j+1]

            if site_energy>=0:
                state[i,j] *= -1
                energy -= 2*site_energy
            else:
                probability = np.exp(2*site_energy/temperature)
                if np.random.random()<probability:
                    state[i,j] *= -1
                    energy -= 2*site_energy
    return energy


def thermalize(size, temperature, boundary_x, boundary_y, ferro, steps=100, init_state=None, plotit=True):
    ''' for square lattice '''
    if init_state is None:
        try:
            state = 1-2*np.floor(np.random.random(size)+0.5)
        except IndexError as err:
            raise ValueError("Wrong input dimension: "+err)

        if len(state.shape)!=2:
            raise ValueError("Wrong input dimension: only two dimension system is allowed")
    else:
        state = init_state.copy()

    # calculate the energy of the initial state
    energy = ferro*state_energy(state, boundary_x, boundary_y)
    energies = [energy]
    # generate new states by flip the spins one by one
    for _ in range(steps):
        energy = update(state, energy, temperature, boundary_x, boundary_y, ferro)
        energies.append(energy)

    if abs(energies[-1]-ferro*state_energy(state, boundary_x, boundary_y))>1e-6:
        raise ValueError("Error when thermalizing: energy not match")

    if plotit:
        plt.plot(energies)
        plt.title('Energies - thermalize for T={:.2f}'.format(temperature))
        plt.show()

    return state, energy


def generate_states(size, temperature, boundary_x=0, boundary_y=0, ferro=1, numbers=200, plotit=True):
    ''' "numbers" is the state number for each temperature '''

    ts = np.repeat(temperature, numbers)
    states = np.zeros([ts.shape[0], np.prod(size)])
    energies = np.zeros(ts.shape)
    begin = time()
    loc = 0
    print('-'*100, end='\r')
    for k, t in enumerate(temperature):
        index = k/len(temperature)*100
        if index>=loc+1:
            loc = int(index)
            time_used = time()-begin
            if time_used<60:
                print('#'*loc + '-'*(100-loc) + " {:.2f} s".format(time_used), end='\r', flush=True)
            elif time_used<3600:
                print('#'*loc + '-'*(100-loc) + " {:.2f} min".format(time_used/60), end='\r', flush=True)
            else:
                print('#'*loc + '-'*(100-loc) + " {:.2f} h   ".format(time_used/3600), end='\r', flush=True)

        state, energy = thermalize(size, t, boundary_x, boundary_y, ferro,
                                    steps=800, init_state=None, plotit=False)
        for i in range(numbers):
            energy = update(state, energy, t, boundary_x, boundary_y, ferro)
            index = k*numbers+i
            energies[index] = energy
            states[index][:] = state.flatten()
    print("#"*100 + " "*10)

    if abs(energies[-1]-ferro*state_energy(state, boundary_x, boundary_y))>1e-6:
        raise ValueError("Error when generating states: energy not match")

    time_used = time()-begin
    if time_used<60:
        print("Time used: {:.2f} s".format(time_used))
    elif time_used<3600:
        print("Time used: {:.2f} min".format(time_used/60))
    else:
        print("Time used: {:.2f} h".format(time_used/3600))

    if plotit:
        plt.plot(energies)
        plt.twinx()
        plt.plot(ts, 'r-')
        plt.title('Energies - generate states')
        plt.show()

    return states


if __name__ == '__main__':
    import argparse
    import Ising_classic
    from inspect import getmembers, isfunction

    parser = argparse.ArgumentParser(description="A collection of properties")
    parser.add_argument('--list', type=bool, nargs='?', const=1, default=0, \
                        help='list all available modules')
    parser.add_argument('--doc', type=str, \
                        help='print documents of given property')
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.list:
        for x in getmembers(Ising_classic, isfunction):
            print(x[0])
        exit()
    if not FLAGS.doc is None:
        print(getattr(Ising_classic, FLAGS.doc).__doc__)
