#!/usr/bin/env python3
# coding: utf-8
r""" A spin-boson model for photoinduced ET reactions in mixed-valence systems
in solution at zero/finite temperature.

Spin-boson model::

    H = H_e + H_v + H_B

where :math:`H_v = 1/2 \sum_j (p_j^2 + \omega_j^2(q_j -\frac{2c_j}{\omega_j^2}
|2><2|)^2)`. H_B similar (discretized from :math:`J_B`)

References
----------
.. [1] J. Chem. Phys. 124, 034114 (2006)
       https://doi.org/10.1063/1.2161178
"""
from __future__ import absolute_import, division, print_function

import logging
from minitn.tensor import Leaf
import os
import sys
import pytest
from builtins import filter, map, range, zip

import numpy as np
from scipy import linalg

from minitn.heom.noise import Correlation, Drude
from minitn.lib.units import Quantity
from minitn.heom.eom import Hierachy
from minitn.heom.propagate import MultiLayer
from minitn.structures.states import Tensor
from minitn.lib.logging import Logger
import pyheom


os.chdir(os.path.abspath(os.path.dirname(__file__)))


def test_brownian():
    lambda_0 = 0.01 # reorganization energy (dimensionless)
    omega_0   = 1.0 # vibrational frequency (dimensionless) 
    zeta      = 0.5 # damping constant      (dimensionless)
    max_tier  = 4
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
    delta = corr_dict['S_delta']

    h = np.array([[omega_1, 0],
                [0, 0]])

    op = np.array([[0, 1],
                [1, 0]])

    max_terms = 3

    corr = Correlation(k_max=max_terms, beta=1)
    corr.symm_coeff = lambda k: s[k, k]    
    corr.asymm_coeff = lambda k: a[k, k]
    corr.exp_coeff = lambda k: gamma[k, k]
    corr.delta_coeff = lambda: delta
    heom = Hierachy([max_tier] * max_terms, h, op, corr)
    rho_0 = np.zeros((2, 2))
    rho_0[0, 0] = 1

    init_wfn = heom.gen_extended_rho(rho_0)

    # Simple HEOM
    root = Tensor(name='root', array=init_wfn, axis=None)
    for k in range(max_terms + 1):
        name_str = k
        l = Leaf(name=name_str)
        root[k] = (l, 1)

    solver = MultiLayer(root, heom.diff(), use_str_name=True)

    solver.settings(
        ode_method='RK45',
        ps_method='s',
        snd_order=False,
    )

    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=5000,
        ode_inter=0.01,
    )):
        if n % 100 == 0:
            rho = np.reshape(r.array, (-1, 4))
            for n, _ in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
    return np.array(dat)



def gen_ref():
    lambda_0 = 0.01 # reorganization energy (dimensionless)
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
        
    dt__unit = 1e-2
            
    rho_0 = np.zeros((n_state,n_state))
    rho_0[0, 0] = 1
    h.set_rho(rho_0)
                
    callback_interval = 5
    count             = 5000

    ref = []
    def callback(t, rho):
        flat_data = [t] + list(np.reshape(rho, -1))
        ref.append(flat_data)
    h.time_evolution(dt__unit, count, callback, callback_interval)
    return np.array(ref)



if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # a1 = gen_ref()
    # np.savetxt('reference.dat', a1)
    a1 = np.loadtxt('reference.dat', dtype=complex)
    a2 = test_brownian()
    np.savetxt('test.dat', a2)
    a2 = np.loadtxt('test.dat', dtype=complex)
    plt.plot(a1[:, 0], a1[:, 1], '-', label='Ikeda')
    plt.plot(a2[:, 0], a2[:, 1], '--', label='minitn')
    plt.legend()
    plt.savefig('cmp.png')

