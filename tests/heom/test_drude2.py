#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.tensor import Tensor
from minitn.heom.network import simple_heom, tensor_train_template

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation, Drude
from minitn.heom.propagate import ProjectorSplitting
from minitn.algorithms.ml import MultiLayer
from minitn.lib.logging import Logger

import pyheom

# Bath
max_tier = 10
max_terms = 3
corr = Drude(lambda_=0.5, omega_0=1.0, k_max=max_terms, beta=1.0)

# System
n_state = 2
epsilon = 1.0
delta = 0.0
H = np.array([[epsilon, delta], [delta, -epsilon]])
V = np.array([[epsilon, 0.0], [0.0, -epsilon]])

# init state
p0 = 1.0
rho_0 = np.array([[p0, 0.0], [0.0, 1.0 - p0]])

dt_unit = 0.001
callback_interval = 1000
count = 500000


def test_simple(fname=None):
    # Type settings
    corr.print()

    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, H, V, corr)

    # Adopt MCTDH
    root = simple_heom(rho_0, n_dims)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    #solver = ProjectorSplitting(root, all_terms)
    solver = MultiLayer(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.atol = 1.e-7
    solver.rtol = 1.e-7

    # Define the obersevable of interest
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
    )):
        try:
            if n % callback_interval == 0:
                rho = np.reshape(r.array, (-1, 4))
                logger.info("{} {} {} {} {}".format(time, rho[0], rho[1], rho[2], rho[3]))
        except:
            break

    return


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'drude_yan2021'))
    prefix = "HEOM_simple_k{}".format(max_terms)

    tst_fname = '{}_tst.dat'.format(prefix)
    # tst_fname = 'test_drude.log'
    ref_fname = '{}_ref.dat'.format(prefix)

    try:
        tst = np.loadtxt(tst_fname, dtype=complex)
    except:
        test_simple(fname=tst_fname)
        tst = np.loadtxt(tst_fname, dtype=complex)

    plt.plot(tst[:, 0], tst[:, 1], '-', label="$P_0$ ({})".format(prefix))
    plt.plot(tst[:, 0], tst[:, -1], '-', label="$P_1$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

    plt.legend()
    plt.title('Drude model (w/ Yan2021)')
    plt.xlim(0, dt_unit * count)
    plt.savefig('{}.png'.format(prefix))
