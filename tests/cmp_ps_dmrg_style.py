#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip

import numpy as np

from minitn.lib.tools import time_this
from sho_model import test_2layers, test_mps_dmrg


@time_this
def ref():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 5, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    for t in exp.root.visitor():
        t.is_normalized = False
    exp.settings(cmf_steps=10, ode_method='RK23')
    t1, a1 = zip(
        *exp.autocorr(steps=100, ode_inter=0.1, fast=True, split=False))
    np.save('./data/ref_t', t1)
    np.save('./data/ref_a', a1)
    return


@time_this
def main2():
    solver = test_mps_dmrg(x0= -5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    solver.settings(cmf_steps=100, ode_method='RK23', ps_method='s',
                    svd_rank=None)
    t1, a1 = zip(*solver.autocorr(steps=100, ode_inter=0.1,
                                  fast=True, split=True))
    np.save('./data/exp2_t', t1)
    np.save('./data/exp2_a', a1)
    return


@time_this
def main():
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    solver.settings(cmf_steps=100, ode_method='RK23', ps_method='u',
                    svd_rank=None)
    t2, a2 = zip(*solver.autocorr(steps=100, ode_inter=0.1,
                                  fast=True, split=True))
    np.save('./data/exp_t', t2)
    np.save('./data/exp_a', a2)


logging.root.setLevel(logging.INFO)
main()
main2()
ref()
