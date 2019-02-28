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
from sho_model import test_2layers, test_mctdh


@time_this
def main():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 2
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    t1, a1 = zip(*exp.autocorr(end=10000, ode_inter=0.001,
                               cmf_step=100, method='RK45'))
    np.save('ml_t', t1)
    np.save('ml_a', a1)
    return


@time_this
def refer():
    x0, x1, n_dvr, n_spf, c = -5., 5., 40, 10, 0.5
    ref = test_mctdh(x0, x1, n_dvr, n_spf, c)
    t2, a2, = zip(*ref.autocorrelation(stop=20., max_inter=0.001))
    np.save('mctdh_t', t2)
    np.save('mctdh_a', a2)
    return


logging.root.setLevel(logging.INFO)
main()
