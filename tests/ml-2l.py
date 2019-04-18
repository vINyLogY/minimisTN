#!/usr/bin/env python
# coding: utf-8
r"""ML type of 4d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip
from functools import partial
from time import time

import numpy as np

from minitn.lib.tools import __, time_this
from sho_model import test_2layers


@time_this
def ref(n_spf=10):
    x0, x1, n_dvr, c, dofs = -5., 5., 40, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    exp.settings(cmf_steps=10, ode_method='RK23')
    t1, a1 = zip(
        *exp.autocorr(steps=1000, ode_inter=0.01, fast=True, split=False))
    np.save('./data/ref_{}_t'.format(n_spf), t1)
    np.save('./data/ref_{}_a'.format(n_spf), a1)
    return


def main():
    logging.root.setLevel(logging.INFO)
    for i in range(2, 20, 4):
        ref(n_spf=i)
    return

main()
