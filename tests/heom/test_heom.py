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
import time
import pyheom


os.chdir(os.path.abspath(os.path.dirname(__file__)))

## Parameters settings
max_tier  = 5

lambda_0 = 0.01 # reorganization energy (dimensionless)
omega_0   = 1.0 # vibrational frequency (dimensionless) 
zeta      = 0.5 # damping constant      (dimensionless)
omega_1 = np.sqrt(omega_0**2 - zeta**2*0.25)
J = pyheom.Brownian(lambda_0, zeta, omega_0)

corr_dict = pyheom.noise_decomposition(
    J,
    T = 1,                      # temperature (dimensionless)
    type_LTC = 'PSD',
    n_PSD = 1,
    type_PSD = 'N-1/N'
)

h = np.array([[omega_1, 0],
                [0, 0]])

op = np.array([[0, 1],
                [1, 0]])

rho_0 = np.zeros((2, 2))
rho_0[0, 0] = 1

dt = 0.01
callback_interval = 5
steps = 5000


def gen_brownian():
    corr = Correlation(k_max=3)
    corr.symm_coeff = lambda k: corr_dict['s'][k, k]    
    corr.asymm_coeff = lambda k: corr_dict['a'][k, k]
    corr.exp_coeff = lambda k: corr_dict['gamma'][k, k]
    corr.delta_coeff = lambda: corr_dict['S_delta']
    heom = Hierachy([max_tier] * corr.k_max, h, op, corr)

    init_wfn = heom.gen_extended_rho(rho_0)

    # Simple HEOM
    root = Tensor(name='root', array=init_wfn, axis=None)
    for k in range(corr.k_max + 1):
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
    data = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=steps,
        ode_inter=dt,
    )):
        if n % callback_interval == 0:
            rho = r.array[0, 0, 0, :]
            flat_data = [time] + list(rho)
            data.append(flat_data)
            print("Time: {}; Pop.: {}".format(time, np.abs(rho[0] + rho[-1])))

    data = np.array(data)
    np.savetxt('tst.dat', data)
    return data


def gen_ref_brownian():
    noises = [
        dict(V=op, C=corr_dict)
    ]

    heom = pyheom.HEOM(
        h,
        noises,
        max_tier=max_tier,
        matrix_type='dense',
        hierarchy_connection='loop',
    )

    heom.set_rho(rho_0)
                
    ref_data = []
    def callback(t, rho):
        flat_data = [t] + list(np.reshape(rho, -1))
        ref_data.append(flat_data)
        print("Time: {}; Pop.: {}".format(t, np.abs(rho[0,0] + rho[1,1])))
    heom.time_evolution(dt, steps, callback, callback_interval)
    
    ref_data = np.array(ref_data)
    np.savetxt('ref.dat', ref_data)
    return ref_data

    
if __name__ == '__main__':
    ref = gen_ref_brownian()
    tst = gen_brownian()
