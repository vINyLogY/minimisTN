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
from minitn.ml import MultiLayer
from sho_model import test_2layers, test_4layers, square, linear


@time_this
def ref():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 8
    solver = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    solver.settings(cmf_steps=10, ode_method='RK45')
    zipped = list(solver.autocorr(steps=500, ode_inter=0.1,
                                  fast=True, split=False))
    np.save('./tmp/ref_mctdh', zipped)
    return


@time_this
def main(split, ps_method='s'):
    solver = test_4layers(x0=-5., x1=5., n_1=5, n_2=5, n_3=5, n_4=40, c=0.5)
    start = time()
    solver.settings(cmf_steps=10, ode_method='RK45', ps_method=ps_method)
    zipped = list(solver.autocorr(steps=500, ode_inter=0.1,
                                  fast=True, split=split))
    end = time()
    print(end - start)
    np.save('./tmp/exp-{}-{}'.format('split' if split else 'origin',
                                     ps_method), zipped)


logging.root.setLevel(logging.INFO)
main(True, 's')
main(False, 's')
ref()
