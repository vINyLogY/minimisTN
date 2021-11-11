#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation, Drude
from minitn.heom.propagate import MultiLayer
from minitn.lib.logging import Logger
from minitn.lib.units import Quantity
from minitn.lib.logging import Logger

# System: pure dephasing
n_state = 2
omega_1 = 0.01
H = np.array([[omega_1, 0.0], [0.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 0.0]])

# init state
rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]])

dt_unit = 0.001
callback_interval = 100
count = 50000

# Bath
# C = g**2 (coth (beta omega / 2)) cos wt - ig**2 sin wt
max_terms = 2
max_tier = 20

omega = 0.05
g = 0.1
beta = 0.1

corr = Correlation(k_max=max_terms)
temp_factor = 1.0 / np.tanh(beta * omega / 2)
corr.symm_coeff = np.array([g**2 * (temp_factor + 1) / 2.0, g**2 * (temp_factor - 1) / 2.0])
corr.asymm_coeff = np.array([0.0, 0.0])
corr.exp_coeff = np.array([1.0j * omega, -1.0j * omega])
corr.delta_coeff = 0.0  # delta_coeff
corr.print()


def test_delta(fname=None):

    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, H, V, corr)

    # Adopt MCTDH
    root = simple_heom(rho_0, n_dims)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

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
            #split=False,
    )):
        if n % callback_interval == 0:
            rho = np.reshape(r.array, (-1, 4))
            logger.info("{} {} {} {} {}".format(time, rho[0, 0], rho[0, 1], rho[0, 2], rho[0, 3]))

    return


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt
    import sys

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'old'))
    prefix = "HEOM_delta_t{}".format(max_tier)

    tst_fname = '{}_tst.dat'.format(prefix)

    try:
        tst = np.loadtxt(tst_fname, dtype=complex)
    except:
        test_delta(fname=tst_fname)
        tst = np.loadtxt(tst_fname, dtype=complex)

    plt.plot(tst[:, 0], np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.abs(tst[:, 2]), '--', label="$|r|$ ({})".format(prefix))

    plt.legend(loc=1)
    plt.title('Delta model')

    plt.savefig('{}.pdf'.format(prefix))
