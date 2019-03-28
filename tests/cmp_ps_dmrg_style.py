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
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    t1, a1 = zip(*solver.autocorr(steps=100, ode_inter=0.01,
                                  method='RK45', fast=False, split=False))
    np.save('./data/ref_t', t1)
    np.save('./data/ref_a', a1)
    return


@time_this
def main():
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    t2, a2 = zip(*solver.autocorr(steps=100, ode_inter=0.01, method='RK45',
                                  fast=False, split=True))
    np.save('./data/exp_t', t2)
    np.save('./data/exp_a', a2)


logging.root.setLevel(logging.INFO)
main()
ref()
