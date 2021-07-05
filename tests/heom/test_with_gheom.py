#!/usr/bin/env python3
# coding: utf-8
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

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'gheom'))


def test_drude():
    from minitn.heom.noise import Drude
    from minitn.lib.units import Quantity

    # System
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au
    # Bath
    lambda_0 = 7.194050 / np.pi     # reorganization energy
    omega_0 = 1.527647e2    # vibrational frequency
    beta = Quantity(300, 'K').value_in_au # temperature
    # Superparameters
    max_terms = 4 # (terms used in the expansion of the correlation function)
    max_tier  = 10 # (number of possble values for each n_k in the extended rho)

    h = np.array([[0, v],
                  [v, e]])

    op = np.array([[0, 0],
                   [0, 1]])

    corr = Drude(lambda_0, omega_0, max_terms, beta)
    print('S: ', corr.symm_coeff)
    print('A: ', corr.asymm_coeff)
    print('gamma: ', corr.exp_coeff)
    print('S_delta: ', corr.delta_coeff)

    coeff = np.loadtxt('coeff.dat')
    sigma, s, a, g = np.transpose(coeff)
    print(sigma * s, sigma * a, g)
    corr2 = Correlation(k_max=max_terms)
    corr2.sysmm_coeff = sigma * s
    corr2.asymm_coeff = sigma * a
    corr2.exp_coeff = g

    def delta_coeff():
        v, l, bh = 1.527647e2, lambda_0, beta
        d = np.sum([(corr2.sysmm_coeff + 1.0j * corr2.asymm_coeff) / corr2.exp_coeff])
        return 2.0 * l / (bh * v) - d

    corr2.delta_coeff = delta_coeff()

    heom = Hierachy([max_tier] * max_terms, h, op, corr)

    phi = [1/np.sqrt(2), 1/np.sqrt(2)] 
    phi /= np.linalg.norm(phi)
    rho_0 = np.tensordot(phi, phi, axes=0)
    init_rho = heom.gen_extended_rho(rho_0)

    # Simple HEOM
    root = Tensor(name='root', array=init_rho, axis=None)
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
                    print('Time: {}; rho: {}{}{}{}'.format(*flat_data))
    return np.array(dat)


if __name__ == '__main__':
    test_drude()

