#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_hseom

from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.seom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import ProjectorSplitting
from minitn.algorithms.ml import MultiLayer
from minitn.lib.units import Quantity

import pyheom

# Bath
eta = 0.05  # reorganization energy (dimensionless)
gamma_c = 0.05  # vibrational frequency (dimensionless)
max_tier = 5
max_terms = 2

J = pyheom.Drudian(eta, gamma_c)
corr_dict = pyheom.noise_decomposition(
    J,
    T=1,  # temperature (dimensionless)
    type_LTC='PSD',
    n_PSD=max_terms - 1,
    type_PSD='N-1/N')

# System
n_state = 2
omega_1 = 0.05
omega_2 = 0.02
H = np.array([[omega_1, omega_2], [omega_2, 0.0]])
V = np.array([[0.0, 0.0], [0.0, 1.0]])

# init state

f0 = np.array([1.0, 0.0])
f1 = np.array([0.0, 1.0])
p = 0.99
init_wfns = np.array([np.sqrt(p) * f0, np.sqrt(1.0 - p) * f1])

dt_unit = 0.001
callback_interval = 100
count = 50000


def test_drude():
    # Type settings
    corr = Correlation(k_max=max_terms)
    corr.symm_coeff = np.diag(corr_dict['s'].toarray())
    corr.asymm_coeff = np.diag(corr_dict['a'].toarray())
    corr.exp_coeff = np.diag(corr_dict['gamma'].toarray())
    corr.delta_coeff = 0.0  # delta_coeff
    corr.print()

    n_dims = [max_tier] * max_terms
    hseom = Hierachy(n_dims, H, V, corr)

    # Adopt MCTDH
    root = simple_hseom(init_wfns, n_dims)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in hseom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = MultiLayer(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.atol = 1.e-7
    solver.rtol = 1.e-7

    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
    )):
        try:
            if n % callback_interval == 0:
                wfns = np.reshape(r.array, (-1, n_state**2))[0]
                print("Time: {};    |c|: {}".format(time, np.linalg.norm(wfns)))

                f = np.reshape(wfns, (n_state, n_state))
                rho = sum(np.outer(i, i) for i in f)
                flat_data = [time] + list(np.reshape(rho, -1))
                dat.append(flat_data)
        except:
            break

    return np.array(dat)


def gen_ref():
    noises = [dict(V=V, C=corr_dict)]
    h = pyheom.HEOM(
        H,
        noises,
        max_tier=max_tier,
        matrix_type='dense',
        hierarchy_connection='loop',
    )

    rho_0 = np.array([[0.5, 0.5], [0.5, 0.5]])

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
    os.chdir(os.path.join(f_dir, 'simple'))
    prefix = "HEOM"

    try:
        tst = np.loadtxt('{}_tst.dat'.format(prefix), dtype=complex)
    except:
        tst = test_drude()
        np.savetxt('{}_tst.dat'.format(prefix), tst)

    # try:
    #     ref = np.loadtxt('{}_ref.dat'.format(prefix), dtype=complex)
    # except:
    #     ref = gen_ref()
    #     np.savetxt('{}_ref.dat'.format(prefix), ref)

    plt.plot(tst[:, 0], tst[:, 1], '-', label="$P_0$ ({})".format(prefix))
    plt.plot(tst[:, 0], tst[:, -1], '-', label="$P_1$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

    # plt.plot(ref[:, 0], ref[:, 1], '--', label="$P_0$ (ref.)".format(prefix))
    # plt.plot(ref[:, 0], ref[:, -1], '--', label="$P_1$ (ref.)".format(prefix))
    # plt.plot(ref[:, 0], np.real(ref[:, 2]), '--', label="$\Re r$ (ref.)".format(prefix))
    # plt.plot(ref[:, 0], np.imag(ref[:, 2]), '--', label="$\Im r$ (ref.)".format(prefix))
    plt.legend()
    plt.title('Drude model w/ pyheom')
    plt.ylim(-0.1, 1.1)
    plt.xlim(0, 10)
    plt.savefig('{}.png'.format(prefix))
