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

data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
os.chdir(data_dir)

def test_diff_heom():
    # System
    e = 0.95
    v = 0
    # Bath
    lambda_0 = 0.01 # reorganization energy
    omega_0 = 1.0 # vibrational frequency
    beta = 1.0 # temperature
    # Superparameters
    max_terms = 2 # (terms used in the expansion of the correlation function)
    max_tier  = 10 # (number of possble values for each n_k in the extended rho)

    h = np.array([[0, v],
                  [v, e]])

    op = np.array([[0, 0],
                   [0, 1]])

    corr = Drude(lambda_0, omega_0, max_terms, beta)
    heom = Hierachy([max_tier] * max_terms, h, op, corr)
    phi = np.array([1.0, 1.0])
    phi /= np.linalg.norm(phi)
    rho_0 = np.tensordot(phi, phi, axes=0)
    init_wfn = heom.gen_extended_rho(rho_0)
    
    # Simple HEOM
    root = Tensor(name='root', array=init_wfn, axis=None)
    for k in range(max_terms + 1):
        name_str = k
        l = Leaf(name=name_str)
        root[k] = (l, 1)

    h_list = heom.diff()
    
    solver = MultiLayer(root, h_list, use_str_name=True)

    solver.settings(
        ode_method='RK45',
        ps_method='s',
        snd_order=False,
    )

    # Define the obersevable of interest
    for time, _ in solver.propagator(
        steps=10,
        ode_inter=0.1,
    ):
        rho = np.reshape(root.array, (-1, 4))[0]
        r = np.reshape(rho, (2,2))
        pr = np.trace(np.dot(r, r))

        flat_data = [time] + list(rho) + [pr]
        print(r'# time    rho00  rho01  rho10  rho11    Pr')
        print('{}    {}  {}  {}  {}    {}'.format(*flat_data))
        print()


if __name__ == '__main__':
    test_diff_heom()
