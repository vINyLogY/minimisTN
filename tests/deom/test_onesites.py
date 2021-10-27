#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.tensor import Tensor
from minitn.heom.network import tensor_train_template

import numpy as np
from minitn.heom.eom import Hierachy
from minitn.heom.noise import Correlation
from minitn.heom.propagate import MultiLayer
from minitn.lib.logging import Logger

# System
n_state = 2
H = np.array([[1.0, 1.0], [1.0, -1.0]])
#H = np.array([[0.0, 0.5], [0.5, 0.0]])
V = np.array([[1.0, 0.0], [0.0, -1.0]])

# Init state
p0 = 1.0
rho_0 = np.array([[p0, 0.0], [0.0, 1.0 - p0]])

# Bath
eta = 0.25  # reorganization energy (dimensionless)
gamma_c = 0.25  # vibrational frequency (dimensionless)
max_tier = 10
max_terms = 3
corr = Correlation(k_max=max_terms)
corr.symm_coeff = np.array([0.4973931166166882, 0.041010943929914466, 0.0765163269642356])
corr.asymm_coeff = np.array([-0.0625, 0.0, 0.0])
corr.exp_coeff = np.array([0.25, 6.305939144224808, 19.499618752922675])
corr.delta_coeff = 0.0  # delta_coeff

dt_unit = 0.01
callback_interval = 100
count = 500000


def test_train(fname=None):
    # HEOM metas
    corr.print()

    n_dims = [max_tier] * max_terms
    heom = Hierachy(n_dims, H, V, corr)

    # 2-site TT
    tensor_train = tensor_train_template(rho_0, n_dims, rank=1)
    root = tensor_train[0]
    leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
    all_terms = []
    for term in heom.diff():
        all_terms.append([(leaves_dict[str(fst)], snd) for fst, snd in term])

    solver = MultiLayer(root, all_terms)
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.svd_err = 1.e-8
    solver.svd_rank = max_tier
    solver.ps_method = 'split'

    projector = np.zeros((max_tier, 1))
    projector[0] = 1.0
    logger = Logger(filename=fname, level='info').logger
    for n, (time, _) in enumerate(solver.propagator(steps=count, ode_inter=dt_unit, split=True)):
        #print('n = {}: '.format(n))
        #for t in tensor_train:
        #    print('{}: {}'.format(t, t.shape))
        if n % callback_interval == 0:
            head = root.array
            for t in tensor_train[1:]:
                spf = Tensor.partial_product(t.array, 1, projector, 0)
                head = Tensor.partial_product(head, head.ndim - 1, spf, 0)

            rho = np.reshape(head, (4, -1))[:, 0]
            logger.info("{} {} {} {} {}".format(time, rho[0], rho[1], rho[2], rho[3]))
    return


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'data'))
    prefix = "HEOM_1site_{:.6f}".format(dt_unit)

    tst_fname = '{}_tst.dat'.format(prefix)
    ref_fname = '{}_ref.dat'.format(prefix)

    tst = test_train(fname=tst_fname)
