#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
from builtins import filter, map, range, zip

import logging

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import MultiLayer
from minitn.heom.network import simple_heom, tensor_train_template
from minitn.tensor import Leaf

import pyheom

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'pyheom'))


def test_drude():
    eta = 0.05 # reorganization energy (dimensionless)
    gamma_c   = 0.05 # vibrational frequency (dimensionless) 
    max_tier  = 5

    max_terms = 4
    J = pyheom.Drudian(eta, gamma_c)
    corr_dict = pyheom.noise_decomposition(
        J,
        T = 1,                      # temperature (dimensionless)
        type_LTC = 'PSD',
        n_PSD = max_terms - 1,
        type_PSD = 'N-1/N'
    )

    s = corr_dict['s'].toarray()
    a = corr_dict['a'].toarray()
    gamma = corr_dict['gamma'].toarray()
    delta = 0

    omega_1 = 0.05
    omega_2 = 0.02
    H = np.array([[omega_1, omega_2],
                [omega_2, 0]])

    V = np.array([[0, 0],
                [0, 1]])

    corr = Correlation(k_max=max_terms, beta=1)
    corr.symm_coeff = np.diag(s)
    corr.asymm_coeff = np.diag(a)
    corr.exp_coeff = np.diag(gamma)
    corr.delta_coeff = delta
    corr.print()
    heom = Hierachy([max_tier] * max_terms, H, V, corr)

    rho_0 = np.zeros((2, 2))
    rho_0[0, 0] = 1

    # Simple HEOM
    root = simple_heom(rho_0, [max_tier] * max_terms)
    #root = tensor_train_template(rho_0, [max_tier] * max_terms, rank=3)
    #root.check_completness(strict=True)

    solver = MultiLayer(root, heom.diff(), use_str_name=True)
    solver.ode_method = 'RK45'
    solver.snd_order = False


    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=20000,
        ode_inter=0.01,
    )):
        if n % 100 == 0:
            rho = np.reshape(r.array, (-1, 4))
            #rho = np.reshape(r.array, (4, -1)).transpose()
            for n, _rn in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
                if n <= 0:
                    print("Time: {}    ; {}:    {}".format(time, n, _rn[0] + _rn[-1]))
    return np.array(dat)



def gen_ref():
    eta = 0.05 # reorganization energy (dimensionless)
    gamma_c   = 0.05 # vibrational frequency (dimensionless) 
    max_tier  = 5

    max_terms = 4
    J = pyheom.Drudian(eta, gamma_c)
    corr_dict = pyheom.noise_decomposition(
        J,
        T = 1,                      # temperature (dimensionless)
        type_LTC = 'PSD',
        n_PSD = max_terms - 1,
        type_PSD = 'N-1/N'
    )

    n_state = 2

    omega_1 = 0.05
    omega_2 = 0.02
    H = np.array([[omega_1, omega_2],
                [omega_2, 0]])

    V = np.array([[0, 0],
                [0, 1]])

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
        
    dt__unit = 0.5e-2
            
    rho_0 = np.zeros((n_state,n_state))
    rho_0[0, 0] = 1
    h.set_rho(rho_0)
                
    callback_interval = 5
    count             = 40000

    ref = []
    def callback(t, rho):
        flat_data = [t] + list(np.reshape(rho, -1))
        ref.append(flat_data)
    h.time_evolution(dt__unit, count, callback, callback_interval)
    return np.array(ref)



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #logging.basicConfig(level=logging.DEBUG)

    # a1 = gen_ref()
    # np.savetxt('reference2.dat', a1)
    a1 = np.loadtxt('reference2.dat', dtype=complex)
    a2 = test_drude()
    np.savetxt('test2.dat', a2)
    # a2 = np.loadtxt('test2.dat', dtype=complex)
    plt.plot(a1[:, 0], a1[:, 1], '-', label='P(0) (pyheom)')
    plt.plot(a2[:, 0], a2[:, 1], '--', label='P(0) (minitn)')
    
    plt.plot(a1[:, 0], a1[:, 4], '-', label='P(1) (pyheom)')
    plt.plot(a2[:, 0], a2[:, 4], '--', label='P(1) (minitn)')
    plt.legend()
    plt.savefig('cmp3.png')
