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
from ft_sho_model import test_2layers


@time_this
def main():
    x0, x1, n_dvr, c, dofs = -5., 5., 40, 0.5, 2
    exp = test_2layers(x0, x1, n_dvr, dofs, c)
    t1, a1 = zip(*exp.autocorr(steps=100, ode_inter=0.01,
                               fast=False, split=True,
                               imaginary=True))
    np.save('ml_t', t1)
    np.save('ml_a', a1)
    x0, x1, n_dvr, c, dofs = -5., 5., 40, 0.5, 2
    exp = test_2layers(x0, x1, n_dvr, dofs, c)
    t1, a1 = zip(*exp.autocorr(steps=100, ode_inter=0.01,
                               fast=False, split=False,
                               imaginary=True))
    for t in exp.root.visitor():
        print(t, np.linalg.matrix_rank(t.array))
    np.save('ml2_t', t1)
    np.save('ml2_a', a1)
    return


logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s', level=logging.DEBUG+1
)
main()
