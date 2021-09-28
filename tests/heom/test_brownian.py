#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import ProjectorSplitting
from minitn.algorithms.ml import MultiLayer
from minitn.lib.units import Quantity
from minitn.lib.logging import Logger

import pyheom

# Bath
lambda_0 = 0.01  # reorganization energy (dimensionless)
omega_0 = 1.0  # vibrational frequency (dimensionless)
zeta = 0.5  # damping constant      (dimensionless)
max_tier = 5
max_terms = 3

J = pyheom.Brownian(lambda_0, zeta, omega_0)
corr_dict = pyheom.noise_decomposition(
    J,
    T=1,  # temperature (dimensionless)
    type_LTC='PSD',
    n_PSD=1,
    type_PSD='N-1/N')

# System
n_state = 2
omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
H = np.array([[omega_1, 0.0], [0.0, 0.0]])
V = np.array([[0.0, 1.0], [1.0, 0.0]])

# init state
rho_0 = np.zeros((n_state, n_state))
rho_0[0, 0] = 1

dt_unit = 0.001
callback_interval = 100
count = 50000

# Type settings
max_terms = 3
corr = Correlation(k_max=max_terms)
corr.symm_coeff = np.diag(corr_dict['s'].toarray())
corr.asymm_coeff = np.diag(corr_dict['a'].toarray())
corr.exp_coeff = np.diag(corr_dict['gamma'].toarray())
corr.delta_coeff = 0.0  # delta_coeff
corr.print()


def test_brownian(fname=None):

    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, H, V, corr)

    # Adopt MCTDH
    root = simple_heom(rho_0, n_dims)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = ProjectorSplitting(root, all_terms)
    #solver = MultiLayer(root, all_terms)
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
        if n % callback_interval == 0:
            rho = np.reshape(r.array, (-1, 4))
            logger.info("{} {} {} {} {}".format(time, rho[0, 0], rho[0, 1], rho[0, 2], rho[0, 3]))
            print("Time: {};    Tr rho_0: {}".format(time, rho[0, 0] + rho[0, -1]))

    return


def gen_ref():
    noises = [dict(V=V, C=corr_dict)]
    h = pyheom.HEOM(
        H,
        noises,
        max_tier=max_tier,
        matrix_type='dense',
        hierarchy_connection='loop',
    )

    h.set_rho(rho_0)

    ref = []

    def callback(t, rho):
        flat_data = [t] + list(np.reshape(rho, -1))
        ref.append(flat_data)

    h.time_evolution(dt_unit, count, callback, callback_interval)
    return np.array(ref)


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'brownian'))
    prefix = "HEOM_brownian"

    tst_fname = '{}_tst.dat'.format(prefix)
    ref_fname = '{}_ref.dat'.format(prefix)

    try:
        tst = np.loadtxt(tst_fname, dtype=complex)
    except:
        test_brownian(fname=tst_fname)
        tst = np.loadtxt(tst_fname, dtype=complex)

    try:
        ref = np.loadtxt(ref_fname, dtype=complex)
    except:
        ref = gen_ref()
        np.savetxt(ref_fname, ref)

    plt.plot(tst[:, 0], np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(prefix))
    #plt.plot(tst[:, 0], np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
    #plt.plot(tst[:, 0], np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

    plt.plot(ref[:, 0], ref[:, 1], '--', label="$P_0$ (ref.)".format(prefix))
    plt.plot(ref[:, 0], ref[:, -1], '--', label="$P_1$ (ref.)".format(prefix))
    #plt.plot(ref[:, 0], np.real(ref[:, 2]), '--', label="$\Re r$ (ref.)".format(prefix))
    #plt.plot(ref[:, 0], np.imag(ref[:, 2]), '--', label="$\Im r$ (ref.)".format(prefix))
    plt.legend(loc=1)
    plt.title('Brownian model w/ pyheom')
    plt.ylim(0, 1.1)
    plt.savefig('{}.png'.format(prefix))
