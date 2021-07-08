#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import MultiLayer
from minitn.structures.states import Tensor
from minitn.tensor import Leaf

import pyheom

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'pyheom'))


def test_brownian():
    lambda_0 = 0.05 # reorganization energy (dimensionless)
    omega_0   = 1.0 # vibrational frequency (dimensionless) 
    zeta      = 0.5 # damping constant      (dimensionless)
    max_tier  = 5
    omega_1 = np.sqrt(omega_0**2 - zeta**2*0.25)

    J = pyheom.Brownian(lambda_0, zeta, omega_0)

    corr_dict = pyheom.noise_decomposition(
        J,
        T = 1,                      # temperature (dimensionless)
        type_LTC = 'PSD',
        n_PSD = 1,
        type_PSD = 'N-1/N'
    )
    s = corr_dict['s'].toarray()
    a = corr_dict['a'].toarray()
    gamma = corr_dict['gamma'].toarray()
    delta = 0

    h = np.array([[omega_1, 0],
                [0, 0]])

    op = np.array([[0, 1],
                [1, 0]])

    max_terms = 3
    corr = Correlation(k_max=max_terms, beta=1)
    corr.symm_coeff = np.diag(s)
    corr.asymm_coeff = np.diag(a)
    corr.exp_coeff = np.diag(gamma)
    corr.delta_coeff = delta
    corr.print()
    heom = Hierachy([max_tier] * max_terms, h, op, corr)
    rho_0 = np.zeros((2, 2))
    rho_0[0, 0] = 1

    init_wfn = heom.gen_extended_rho(rho_0)


    solver = MultiLayer(init_wfn, heom.diff())


    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=5000,
        ode_inter=0.01,
    )):
        if n % 100 == 0:
            rho = np.reshape(r, (-1, 4))
            for n, _rn in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
                if n <= 0:
                    print("Time: {}    ; {}:    {}".format(time, n, _rn[0] + _rn[-1]))
    return np.array(dat)



def gen_ref():
    lambda_0 = 0.05 # reorganization energy (dimensionless)
    omega_0   = 1.0 # vibrational frequency (dimensionless) 
    zeta      = 0.5 # damping constant      (dimensionless)
    max_tier  = 5

    J = pyheom.Brownian(lambda_0, zeta, omega_0)
    corr_dict = pyheom.noise_decomposition(
        J,
        T = 1,                      # temperature (dimensionless)
        type_LTC = 'PSD',
        n_PSD = 1,
        type_PSD = 'N-1/N'
    )

    n_state = 2

    omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
    H = np.array([[omega_1, 0],
                [0, 0]])

    V = np.array([[0, 1],
                [1, 0]])

    noises = [
        dict(V=V, C=corr_dict)
    ]

    h = pyheom.HEOM(
        H,
        noises,
        max_tier=max_tier,
        matrix_type='dense',
        hierarchy_connection='loop',
    )
        
    dt__unit = 5.0e-3
            
    rho_0 = np.zeros((n_state,n_state))
    rho_0[0, 0] = 1
    h.set_rho(rho_0)
                
    callback_interval = 5
    count             = 10000

    ref = []
    def callback(t, rho):
        flat_data = [t] + list(np.reshape(rho, -1))
        ref.append(flat_data)
    h.time_evolution(dt__unit, count, callback, callback_interval)
    return np.array(ref)



if __name__ == '__main__':
    from matplotlib import pyplot as plt

    #a1 = gen_ref()
    #np.savetxt('reference.dat', a1)
    a1 = np.loadtxt('reference.dat', dtype=complex)
    a2 = test_brownian()
    np.savetxt('test.dat', a2)
    #a2 = np.loadtxt('test.dat', dtype=complex)
    plt.plot(a1[:, 0], a1[:, 1], '-', label='Ikeda')
    plt.plot(a2[:, 0], a2[:, 1], '--', label='minitn')
    plt.legend()
    plt.savefig('cmp2.png')
