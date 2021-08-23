#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import ProjectorSplitting
from minitn.lib.units import Quantity

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'networks'))


def test_drude():
    # System
    n_states = 2
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(0, 'cm-1').value_in_au
    h = np.array([[e, v], [v, 0]])
    op = np.array([[1, 0], [0, -1]])

    # Bath
    max_terms = 1
    corr = Correlation(k_max=1)
    corr.symm_coeff = np.array([4.66691921e+01 * 9.24899189e+01])
    corr.asymm_coeff = np.array([4.66691921e+01 * -2.35486582e+01])
    corr.exp_coeff = np.array([1.0])
    corr.delta_coeff = 0.0  # delta_coeff
    corr.print()

    # Superparameters
    max_tier = 5  # (number of possble values for each n_k in the extended rho)
    n_shape = [max_tier] * max_terms
    heom = Hierachy([max_tier], h, op, corr)

    # inital rho_n
    rho_0 = np.zeros((2, 2))
    rho_0[0, 0] = 1

    # Adopt MCTDH
    root = simple_heom(rho_0, n_shape)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = ProjectorSplitting(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.atol = 1.e-7
    solver.rtol = 1.e-7

    # Define the obersevable of interest
    dat = []
    for n, (time,
            r) in enumerate(solver.propagator(
                steps=50000,
                ode_inter=0.01,
            )):
        try:
            if n % 100 == 0:
                rho = np.reshape(r.array, (-1, 4))
                for n, _rn in enumerate(rho):
                    if n == 0:
                        flat_data = [time] + list(rho[0])
                        dat.append(flat_data)
                        print("Time: {};    Tr rho_{}: {}".format(
                           time, n, _rn[0] + _rn[-1]))
        except:
            break
        
    return np.array(dat)


def test_brownian():
    # system
    omega_0 = 1.0  # vibrational frequency (dimensionless)
    zeta = 0.5  # damping constant      (dimensionless)
    omega_1 = np.sqrt(omega_0**2 - zeta**2 * 0.25)
    h = np.array([[omega_1, 0], [0, 0]])

    # bath-interaction
    op = np.array([[0, 1], [1, 0]])

    max_terms = 3
    corr = Correlation(k_max=max_terms, beta=1)
    corr.symm_coeff = np.array([
        0.01082299 + 0.00237646j, 0.01082299 - 0.00237646j, -0.00010451 + 0.j
    ])
    corr.asymm_coeff = np.array([
        0. - 0.00516398j, 0. + 0.00516398j, -0.00010451 + 0.j, 0. + 0.j
    ])
    corr.exp_coeff = np.array([
        0.25 + 0.96824584j, 0.25 - 0.96824584j, 7.74596669 + 0.j
    ])
    corr.delta_coeff = 0.0  # delta_coeff
    corr.print()

    # Superparameters
    max_tier = 5  # (number of possble values for each n_k in the extended rho)
    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, h, op, corr)

    # inital rho_n
    rho_0 = np.zeros((2, 2))
    rho_0[0, 0] = 1

    # Adopt MCTDH
    root = simple_heom(rho_0, n_dims)
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = ProjectorSplitting(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.atol = 1.e-7
    solver.rtol = 1.e-7

    # Define the obersevable of interest
    dat = []
    for n, (time,
            r) in enumerate(solver.propagator(
                steps=5000,
                ode_inter=0.01,
            )):
        if n % 100 == 0:
            rho = np.reshape(r.array, (-1, 4))
            for n, _rn in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
                if n <= 0:
                    print("Time: {};    Tr rho_{}: {}".format(
                        time, n, _rn[0] + _rn[-1]))
    return np.array(dat)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    prefix = "HEOM"
    a = test_drude()
    np.savetxt('test_simple.dat'.format(prefix), a)
    #a = np.loadtxt('test_simple.dat'.format(prefix), dtype=complex)
    plt.plot(a[:, 0], a[:, 1], '-', label="$P_0$ ({})".format(prefix))
    plt.plot(a[:, 0], a[:, -1], '-', label="$P_1$ ({})".format(prefix))
    plt.plot(a[:, 0],
             np.real(a[:, 2]),
             '-',
             label="$\Re r$ ({})".format(prefix))
    plt.plot(a[:, 0],
             np.imag(a[:, 2]),
             '-',
             label="$\Im r$ ({})".format(prefix))
    plt.legend()
    plt.savefig('{}.png'.format(prefix))
