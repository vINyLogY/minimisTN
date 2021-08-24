#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.propagate import ProjectorSplitting

import os
from builtins import filter, map, range, zip

import logging

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.algorithms.ml import MultiLayer
from minitn.heom.network import simple_heom, tensor_train_template
from minitn.tensor import Leaf, Tensor

import pyheom

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'tensor_train'))


def test_drude_train():
    eta = 0.05  # reorganization energy (dimensionless)
    gamma_c = 0.05  # vibrational frequency (dimensionless)
    max_tier = 10

    max_terms = 3
    J = pyheom.Drudian(eta, gamma_c)
    corr_dict = pyheom.noise_decomposition(
        J,
        T=1,  # temperature (dimensionless)
        type_LTC='PSD',
        n_PSD=max_terms - 1,
        type_PSD='N-1/N')

    s = corr_dict['s'].toarray()
    a = corr_dict['a'].toarray()
    gamma = corr_dict['gamma'].toarray()
    delta = 0

    omega_1 = 0.05
    omega_2 = 0.02
    H = np.array([[omega_1, omega_2], [omega_2, 0]])

    V = np.array([[0, 0], [0, 1]])

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
    tensor_train = tensor_train_template(rho_0, [max_tier] * max_terms, rank=max_tier)
    root = tensor_train[0]
    #root.check_completness(strict=True)

    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = ProjectorSplitting(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.max_ode_steps = 100000

    # Define the obersevable of interest
    projector = np.zeros((max_tier, 1))
    projector[0] = 1.0

    dat = []
    for n, (time, r) in enumerate(solver.propagator(
            steps=20000,
            ode_inter=0.01,
    )):
        if n % 2 == 0:

            head = root.array
            for t in tensor_train[1:]:
                head = Tensor.partial_product(head, head.ndim - 1, t.array, 0)

            print(head.shape)

            rho = np.reshape(head, (4, -1))[:, 0]
            flat_data = [time] + list(rho)
            dat.append(flat_data)
            print("Time {} | Pop_1 {} | Total {}".format(time, rho[0], rho[0] + rho[-1]))

            # Try
            head2 = root.array
            for t in tensor_train[1:]:
                spf = Tensor.partial_product(t.array, 1, projector, 0)
                head2 = Tensor.partial_product(head2, head2.ndim - 1, spf, 0)

            rho2 = np.reshape(head2, (4, -1))[:, 0]
            print(np.allclose(rho2, rho))

    return np.array(dat)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    #logging.basicConfig(level=logging.DEBUG)

    a2 = test_drude_train()
    np.savetxt('test2.dat', a2)
    #a2 = np.loadtxt('test2.dat', dtype=complex)
    plt.plot(a2[:, 0], a2[:, 1], '--', label='P(0) (minitn)')
    plt.plot(a2[:, 0], a2[:, 4], '--', label='P(1) (minitn)')
    plt.legend()
    plt.savefig('cmp.png')
