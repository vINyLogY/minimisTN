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
def main1():
    # x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 4
    exp = test_2layers(lower=-10, upper=10, dofs=2, n_dvr=100, n_spf=5,
                       random_seed=233)
    exp.settings(cmf_steps=10,
                 ode_method='RK23',
                 ps_method='s',
                 svd_rank=5,
                 snd_order=False)
    t1, a1 = zip(*exp.autocorr(steps=300,
                               ode_inter=0.01,
                               fast=True,
                               split=True))
    np.save('./exp_t', t1)
    np.save('./exp_a', a1)
    return


@time_this
def main2():
    # x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 6, 0.5, 4
    exp = test_2layers(lower=-10, upper=10, dofs=2, n_dvr=100, n_spf=5,
                       random_seed=233)
    exp.settings(cmf_steps=10,
                 ode_method='RK23',
                 ps_method='u',
                 svd_rank=5,
                 snd_order=False)
    t1, a1 = zip(*exp.autocorr(steps=300,
                               ode_inter=0.01,
                               fast=True,
                               split=True))
    np.save('./ref_t', t1)
    np.save('./ref_a', a1)
    return


@time_this
def refer():
    x0, x1, n_dvr, n_spf, dof, c = -5., 5., 40, 10, 4, 0.5
    ref = test_mctdh(x0, x1, n_dvr, n_spf, dof, c)
    t2, a2, = zip(*ref.autocorrelation(stop=20., max_inter=0.01))
    np.save('mctdh_t', t2)
    np.save('mctdh_a', a2)
    return


if __name__ == '__main__':
    logging.basicConfig(
        format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s',
        level=logging.DEBUG
    )
    main1()
    main2()
    # refer()
