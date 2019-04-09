#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division


import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from minitn.lib.tools import __, time_this, figure
from ft_sho_model import test_2layers


@time_this
def main():
    exp = test_2layers()
    exp.settings(cmf_steps=10,
                 ode_method='RK45',
                 ps_method='s')
    t1, a1 = zip(*exp.autocorr(steps=100, ode_inter=0.1,
                               fast=False, split=True,
                               imaginary=True))
    np.save('ml_t', t1)
    np.save('ml_a', a1)
    exp = test_2layers()
    exp.settings(cmf_steps=10,
                 ode_method='RK45')
    t1, a1 = zip(*exp.autocorr(steps=100, ode_inter=0.1,
                               fast=False, split=False,
                               imaginary=True))
    for t in exp.root.visitor():
        print(t, np.linalg.matrix_rank(t.array))
    np.save('ml2_t', t1)
    np.save('ml2_a', a1)
    return


def ref(c=0.5, n_dvr=40, dofs=2):
    logging.info('Reference')
    po = np.identity(dofs)
    po += c * np.eye(dofs, k=1)
    po += c * np.eye(dofs, k=-1)
    w = np.sqrt(linalg.eigh(po, eigvals_only=True))
    for n in range(100):
        beta = n * 0.2
        ex = beta * w
        exp = np.exp

        def _z(x): return (
            (1. - exp(-n_dvr * x)) / (exp(0.5 * x) - exp(-0.5 * x))
        )
        z = np.prod(list(map(_z, ex)))
        logging.info('beta: {:.3f}; Z: {}'.format(beta, z))
    return


logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s', level=logging.DEBUG+1
)
ref()
main()
