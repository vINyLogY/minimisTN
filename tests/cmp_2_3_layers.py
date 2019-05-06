#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip
from functools import partial
from time import time
import sys

import numpy as np

from minitn.lib.tools import __, time_this, figure
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR
from minitn.mctdh import MCTDH
from minitn.ml import MultiLayer
from sho_model import test_2layers, square, linear, triangular


@time_this
def ref():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    for t in exp.root.visitor():
        t.is_normalized = False
    exp.settings(cmf_steps=10, ode_method='RK23')
    t1, a1 = zip(
        *exp.autocorr(steps=100, ode_inter=0.1, fast=True, split=False))
    np.save('./data/ref_t', t1)
    np.save('./data/ref_a', a1)
    return


# split method 1: split-unite
@time_this
def main():
    graph = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7],
        4: [8],
        5: [9],
        6: [10]
    }
    n_1, n_2, n_3 = 10, 10, 40
    dvr = SineDVR(-5., 5, n_3)
    dvr.set_v_func(square)
    _, array_i = dvr.solve(n_state=n_2)

    root = Tensor.generate(graph, 0)
    leaves = []
    for t in root.visitor():
        t.is_normalized = False
        if int(t.name) == 0:
            array = np.zeros((n_1, n_1))
            array[0, 0] = 1.
        elif 1 <= int(t.name) <= 2:
            array = np.zeros((n_1, n_2 ** 2))
            for n, v_i in zip(triangular(n_2), array):
                v_i[n] = 1.
            array = np.reshape(array, (n_1, n_2, n_2))
        elif 3 <= int(t.name) <= 6:
            array = array_i
        else:
            array = None
            leaves.append(t)
        t.set_array(array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    # single
    s_h = dvr.h_mat()
    for leaf in leaves:
        h_list.append([(leaf, s_h)])
    # couple
    linear_ = partial(linear, c=0.5)
    dvr.set_v_func(linear_)
    l_h = dvr.v_mat()
    for i in range(4 - 1):
        term = [(leaves[i], l_h), (leaves[i + 1], l_h)]
        h_list.append(term)

    for t in root.visitor():
        t.check_completness(strict=True)

    solver = MultiLayer(root, h_list)
    solver.settings(cmf_steps=10, ode_method='RK23', ps_method='u',
                    svd_rank=8)
    start = time()
    t2, a2 = zip(*
        solver.autocorr(steps=100, ode_inter=0.1, fast=True, split=True))
    end = time()
    print(end - start)
    np.save('./data/exp_t', t2)
    np.save('./data/exp_a', a2)


# split method 2: unite-split
@time_this
def main2():
    graph = {
        0: [1, 2],
        1: [3, 4],
        2: [5, 6],
        3: [7],
        4: [8],
        5: [9],
        6: [10]
    }
    n_1, n_2, n_3 = 10, 10, 40
    dvr = SineDVR(-5., 5, n_3)
    dvr.set_v_func(square)
    _, array_i = dvr.solve(n_state=n_2)

    root = Tensor.generate(graph, 0)
    leaves = []
    for t in root.visitor():
        t.is_normalized = False
        if int(t.name) == 0:
            array = np.zeros((n_1, n_1))
            array[0, 0] = 1.
        elif 1 <= int(t.name) <= 2:
            array = np.zeros((n_1, n_2 ** 2))
            for n, v_i in zip(triangular(n_2), array):
                v_i[n] = 1.
            array = np.reshape(array, (n_1, n_2, n_2))
        elif 3 <= int(t.name) <= 6:
            array = array_i
        else:
            array = None
            leaves.append(t)
        t.set_array(array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    # single
    s_h = dvr.h_mat()
    for leaf in leaves:
        h_list.append([(leaf, s_h)])
    # couple
    linear_ = partial(linear, c=0.5)
    dvr.set_v_func(linear_)
    l_h = dvr.v_mat()
    for i in range(4 - 1):
        term = [(leaves[i], l_h), (leaves[i + 1], l_h)]
        h_list.append(term)

    for t in root.visitor():
        t.check_completness(strict=True)

    solver = MultiLayer(root, h_list)
    solver.settings(cmf_steps=10, ode_method='RK23', ps_method='s',
                    svd_rank=8)
    start = time()
    t2, a2 = zip(*
        solver.autocorr(steps=100, ode_inter=0.1, fast=True, split=True))
    end = time()
    print(end - start)
    np.save('./data/exp2_t', t2)
    np.save('./data/exp2_a', a2)


logging.basicConfig(
    format='(In %(module)s)[%(funcName)s] %(message)s',
    stream=sys.stderr, level=logging.DEBUG
)
# main()
# main2()
ref()
