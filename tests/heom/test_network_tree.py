#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

import os
from builtins import filter, map, range, zip

import logging

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.algorithms.ml import MultiLayer
from minitn.heom.network import simple_heom, tensor_tree_template
from minitn.tensor import Leaf, Tensor

import pyheom

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'tensor_train'))


def test_drude_tree():
    eta = 0.05 # reorganization energy (dimensionless)
    gamma_c   = 0.05 # vibrational frequency (dimensionless) 
    max_tier  = 10

    max_terms = 3
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

    root = tensor_tree_template(rho_0, [max_tier] * max_terms, rank=max_tier//2)

    solver = MultiLayer(root, heom.diff(), use_str_name=True)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.max_ode_steps = 100000

    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=20000,
        ode_inter=0.01,
    )):
        if n % 100 == 0:
            
            head = root.array

            print(head.shape)

            rho = Tensor.partial_product(r.array, 0, r[0][0].array, 0)
            rho = np.reshape(rho, (-1, 4))

            flat_data = [time] + list(rho[0])
            dat.append(flat_data)
            print("Time: {} | Pop 0: {} | Total: {}".format(flat_data[0], flat_data[1], flat_data[1] + flat_data[-1]))

    return np.array(dat)


if __name__ == '__main__':
    #logging.basicConfig(level=logging.DEBUG)

    a2 = test_drude_tree()
