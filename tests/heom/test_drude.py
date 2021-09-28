#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.tensor import Tensor
from minitn.heom.network import simple_heom, tensor_train_template

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import ProjectorSplitting
from minitn.algorithms.ml import MultiLayer
from minitn.lib.logging import Logger

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
p0 = 1.0
rho_0 = np.array([[p0, 0.0], [0.0, 1.0 - p0]])

dt_unit = 0.001
callback_interval = 1000
count = 500000


def test_simple():
    # Type settings
    corr = Correlation(k_max=max_terms)
    corr.symm_coeff = np.diag(corr_dict['s'].toarray())
    corr.asymm_coeff = np.diag(corr_dict['a'].toarray())
    corr.exp_coeff = np.diag(corr_dict['gamma'].toarray())
    corr.delta_coeff = 0.0  # delta_coeff
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
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
    )):
        try:
            if n % callback_interval == 0:
                rho = np.reshape(r.array, (-1, 4))
                flat_data = [time] + list(rho[0])
                dat.append(flat_data)
                print("Time: {};    Tr rho_0: {}".format(time, rho[0, 0] + rho[0, -1]))
        except:
            break

    return np.array(dat)


def test_train(fname=None):
    # Type settings
    corr = Correlation(k_max=max_terms)
    corr.symm_coeff = np.diag(corr_dict['s'].toarray())
    corr.asymm_coeff = np.diag(corr_dict['a'].toarray())
    corr.exp_coeff = np.diag(corr_dict['gamma'].toarray())
    corr.delta_coeff = 0.0  # delta_coeff
    corr.print()

    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, H, V, corr)

    # Adopt TT
    tensor_train = tensor_train_template(rho_0, n_dims)
    root = tensor_train[0]
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = MultiLayer(root, all_terms)
    #solver = ProjectorSplitting(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.atol = 1.e-7
    solver.rtol = 1.e-7
    solver.ps_method = 'split-unite'

    projector = np.zeros((max_tier, 1))
    projector[0] = 1.0
    logger = Logger(filename=fname, level='info').logger
    for n, (time, _) in enumerate(solver.propagator(steps=count, ode_inter=dt_unit, split=False)):
        if n % callback_interval == 0:
            head = root.array
            for t in tensor_train[1:]:
                spf = Tensor.partial_product(t.array, 1, projector, 0)
                head = Tensor.partial_product(head, head.ndim - 1, spf, 0)

            rho = np.reshape(head, (4, -1))[:, 0]
            logger.info("{} {} {} {} {}".format(time, rho[0], rho[1], rho[2], rho[3]))
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
    os.chdir(os.path.join(f_dir, 'drude'))
    prefix = "HEOM_TT"

    tst_fname = '{}_tst.dat'.format(prefix)
    # tst_fname = 'test_drude.log'
    ref_fname = '{}_ref.dat'.format(prefix)

    try:
        tst = np.loadtxt(tst_fname, dtype=complex)
    except:
        #tst = test_simple()
        tst = test_train(fname=tst_fname)
        tst = np.loadtxt(tst_fname, dtype=complex)

    # generate reference data from pyheom
    ref = gen_ref()
    np.savetxt(ref_fname, ref)

    plt.plot(tst[:, 0], tst[:, 1], '-', label="$P_0$ ({})".format(prefix))
    plt.plot(tst[:, 0], tst[:, -1], '-', label="$P_1$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
    plt.plot(tst[:, 0], np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

    plt.plot(ref[:, 0], ref[:, 1], '--', label="$P_0$ (ref.)".format(prefix))
    plt.plot(ref[:, 0], ref[:, -1], '--', label="$P_1$ (ref.)".format(prefix))
    plt.plot(ref[:, 0], np.real(ref[:, 2]), '--', label="$\Re r$ (ref.)".format(prefix))
    plt.plot(ref[:, 0], -np.imag(ref[:, 2]), '--', label="$\Im r$ (ref.)".format(prefix))

    plt.legend()
    plt.title('Drude model (TT)')
    plt.xlim(0, dt_unit * count)
    plt.savefig('{}.png'.format(prefix))
