#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom, simple_hseom

import os
from builtins import filter, map, range, zip

import numpy as np
from minitn.heom.seom import Hierachy
from minitn.heom.noise import Correlation
from minitn.algorithms.ml import MultiLayer
from minitn.lib.units import Quantity

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'drude'))


def test_drude():
    # System
    n_states = 2
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au
    h = np.array([[e, v], [v, 0]])
    op = np.array([[1, 0], [0, -1]])

    # Bath
    max_terms = 1
    corr = Correlation(k_max=1)
    corr.symm_coeff = np.array([4.66691921e+01 * 9.24899189e+01])
    corr.asymm_coeff = np.array([4.66691921e+01 * -2.35486582e+01])
    corr.exp_coeff = np.array([1.0])
    corr.delta_coeff = 0.0  # delta_coeff()
    corr.print()

    # Superparameters
    max_tier = 10  # (number of possble values for each n_k in the extended rho)
    n_shape = [max_tier] * max_terms
    heom = Hierachy([max_tier], h, op, corr)
    heom.inv_ihbar = 1.0

    # inital wfn
    wfn = np.array([1.0, 0.0], dtype=np.complex128)

    # Adopt MCTDH
    root = simple_hseom(wfn, n_shape)
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
    dat = []
    for n, (time,
            r) in enumerate(solver.propagator(
                steps=50000,
                ode_inter=0.01,
            )):
        if n % 100 == 0:
            spf = r[max_terms][0].array
            wfn = np.reshape(r.array, (np.prod(n_shape), -1))[0]
            rho = np.conj(np.transpose(spf)) @ np.diag(np.abs(wfn)**2) @ spf
            flat_data = [time] + list(np.reshape(rho, -1))
            dat.append(flat_data)
            print("Time: {};  Pop: {};  Trace: {}".format(
                time, flat_data[1], flat_data[1] + flat_data[-1]))
    return np.array(dat)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    prefix = "HSEOM"
    a = test_drude()
    np.savetxt('test_simple.dat'.format(prefix), a)
    #a = np.loadtxt('test_simple.dat'.format(prefix), dtype=complex)
    plt.plot(a[:, 0], a[:, 1], '-', label=prefix)
    plt.legend()
    plt.savefig('{}.png'.format(prefix))
