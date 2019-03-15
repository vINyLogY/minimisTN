#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip
from functools import partial
from time import time

import numpy as np

from minitn.lib.tools import __, time_this, figure
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR
from minitn.mctdh import MCTDH
from minitn.ml import MultiLayer
from sho_model import test_2layers, test_mps_dmrg


@time_this
def ref():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 6, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    t1, a1 = zip(*exp.autocorr(steps=10000, ode_inter=0.001, cmf_step=100,
                               method='RK45', fast=True, split=False))
    np.save('./data/ref_t', t1)
    np.save('./data/ref_a', a1)
    return


@time_this
def main():
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    start = time()
    t2, a2 = zip(*solver.autocorr(steps=10000, ode_inter=0.001, cmf_step=100,
                                  method='RK45', fast=True, split=False))
    end = time()
    print(end - start)
    np.save('./data/exp_t', t2)
    np.save('./data/exp_a', a2)


logging.root.setLevel(logging.DEBUG)
main()
ref()
