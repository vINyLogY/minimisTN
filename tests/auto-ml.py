#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division


import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from minitn.lib.tools import __, time_this, figure
from sho_model import test_2layers, test_mps_dmrg, test_mctdh


@time_this
def main():
    # x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 4
    exp = test_2layers(dofs=3, n_spf=10)
    exp.settings(cmf_steps=10,
                 ode_method='RK23',
                 ps_method='s',
                 svd_rank=13)
    t1, a1 = zip(*exp.autocorr(steps=30,
                               ode_inter=0.1,
                               fast=False,
                               split=True))
    np.save('data/exp_t', t1)
    np.save('data/exp_a', a1)
    # x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 6, 0.5, 4
    exp = test_2layers(dofs=3, n_spf=10)
    exp.settings(cmf_steps=10,
                 ode_method='RK23')
    t1, a1 = zip(*exp.autocorr(steps=30,
                               ode_inter=0.1,
                               fast=False,
                               split=False))
    np.save('data/ref_t', t1)
    np.save('data/ref_a', a1)
    return


@time_this
def refer():
    x0, x1, n_dvr, n_spf, dof, c = -5., 5., 40, 10, 4, 0.5
    ref = test_mctdh(x0, x1, n_dvr, n_spf, dof, c)
    t2, a2, = zip(*ref.autocorrelation(stop=20., max_inter=0.01))
    np.save('mctdh_t', t2)
    np.save('mctdh_a', a2)
    return


logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s',
    level=logging.DEBUG
)
main()
# refer()
