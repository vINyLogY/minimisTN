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
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    start = time()
    t1, a1 = zip(*solver.autocorr(steps=100, ode_inter=0.1, cmf_steps=1,
                                  method='RK45', fast=False, split=False))
    end = time()
    print(end - start)
    np.save('./data/ref_t', t1)
    np.save('./data/ref_a', a1)
    return


@time_this
def main():
    solver = test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5)
    start = time()
    t2, a2 = zip(*solver.autocorr(steps=100, ode_inter=0.1, method='RK45',
                                  fast=False, split=True))
    end = time()
    print(end - start)
    np.save('./data/exp_t', t2)
    np.save('./data/exp_a', a2)


logging.root.setLevel(logging.INFO)
main()
ref()
