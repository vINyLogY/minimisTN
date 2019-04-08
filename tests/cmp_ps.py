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
from sho_model import test_2layers, test_4layers, test_mctdh


@time_this
def main():
    x0, x1, n_dvr, n_spf, c, dofs = -10., 10., 50, 5, 0.5, 3
    exp1 = test_2layers(x0, x1, n_dvr, n_spf, dofs, c,
                        random_seed=None)
    exp1 = test_4layers()
    exp1.settings(cmf_steps=100)
    g1 = exp1.autocorr(steps=100, ode_inter=0.01, split=True)
    exp2 = test_2layers(x0, x1, n_dvr, n_spf, dofs, c,
                        random_seed=None)
    exp2 = test_4layers()
    exp2.settings(cmf_steps=10)
    g2 = exp2.autocorr(steps=100, ode_inter=0.01, split=False)
    if __debug__:
        v1, v2 = exp1.root.vectorize(), exp2.root.vectorize()
        assert np.allclose(np.abs(v1), np.abs(v2))
    (t1, a1), (t2, a2) = zip(*g1), zip(*g2)
    with figure():
        plt.plot(t2, np.real(a2), '-')
        plt.plot(t1, np.real(a1), '--')
        plt.show()
    with figure():
        plt.plot(t2, np.imag(a2), '-')
        plt.plot(t1, np.imag(a1), '--')
        plt.show()
    with figure():
        plt.plot(t2, np.abs(a2), '-')
        plt.plot(t1, np.abs(a1), '--')
        plt.show()
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
    level=logging.INFO
)
main()
# refer()
