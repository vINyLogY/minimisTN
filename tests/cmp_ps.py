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
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 6, 0.5, 2
    exp1 = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    g1 = exp1.autocorr(steps=500, ode_inter=0.01, cmf_step=None,
                       method='RK23', fast=False, split=True)
    r1 = exp1.root
    exp2 = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    g2 = exp2.autocorr(steps=500, ode_inter=0.01, cmf_step=None,
                       method='RK23', fast=False, split=False)
    r2 = exp2.root
    for (t1, _), (t2, _) in zip(g1, g2):
        assert(t1 == t2)
        # v1 = r1.vectorize()
        # v2 = r2.vectorize()
        # d = np.abs(v1)-np.abs(v2)
        # for n, i in enumerate(d):
        #     if abs(i) > 1.e-14:
        #         print('n: {}, i: {}'.format(n, i))
        # pass
    return


@time_this
def refer():
    x0, x1, n_dvr, n_spf, dof, c = -5., 5., 40, 10, 2, 0.5
    ref = test_mctdh(x0, x1, n_dvr, n_spf, dof, c)
    t2, a2, = zip(*ref.autocorrelation(stop=20., max_inter=0.001))
    np.save('mctdh_t', t2)
    np.save('mctdh_a', a2)
    return

logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s',
    level=logging.DEBUG
)
main()
# refer()
