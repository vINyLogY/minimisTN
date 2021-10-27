#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from minitn.heom.network import simple_heom

from minitn.lib.backend import np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import MultiLayer
from minitn.lib.logging import Logger

# Bath
eta = 0.25  # reorganization energy (dimensionless)
gamma_c = 0.25  # vibrational frequency (dimensionless)
max_tier = 10
max_terms = 3

# System
n_state = 2
H = np.array([[1.0, 1.0], [1.0, -1.0]])
#H = np.array([[0.0, 0.5], [0.5, 0.0]])
V = np.array([[1.0, 0.0], [0.0, -1.0]])

# init state
p0 = 1.0
rho_0 = np.array([[p0, 0.0], [0.0, 1.0 - p0]])

corr = Correlation(k_max=max_terms)
corr.symm_coeff = np.array([0.4973931166166882, 0.041010943929914466, 0.0765163269642356])
corr.asymm_coeff = np.array([-0.0625, 0.0, 0.0])
corr.exp_coeff = np.array([0.25, 6.305939144224808, 19.499618752922675])
corr.delta_coeff = 0.0  # delta_coeff

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

    # Define the obersevable of interest
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
    )):
        try:
            if n % callback_interval == 0:
                rho = np.reshape(r.array, (-1, 4))[0]
                logger.info("{} {} {} {} {}".format(time, rho[0], rho[1], rho[2], rho[3]))
        except:
            break

    return


if __name__ == '__main__':
    import os
    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'data'))
    prefix = "HEOM_simple_t{}".format(max_tier)

    tst_fname = '{}_tst.dat'.format(prefix)
    ref_fname = '{}_ref.dat'.format(prefix)

    tst = test_simple(fname=tst_fname)
