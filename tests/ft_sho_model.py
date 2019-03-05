#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.lib.tools import __, time_this, figure
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR
from minitn.mctdh import MCTDH
from minitn.ml import MultiLayer


def square(x): return 0.5 * (x ** 2)


def linear(x, c=0.5): return c * x


def test_2layers(lower, upper, n_dvr, n_spf, dofs, c):
    assert(n_spf < n_dvr)

    # Create a graph
    root = Tensor(name='S')
    basis = []
    hamiltonian = []
    for i in range(2 * dofs):
        bi = Tensor(name='B' + str(i), axis=1)
        basis.append(bi)
        hi = Leaf(name='L' + str(i))
        hamiltonian.append(hi)
        bi.link_to(1, root, i)
        bi.link_to(0, hi, 0)

    # Generate initial state
    dvr = SineDVR(lower, upper, n_dvr)
    dvr.set_v_func(square)
    # SPFs
    array_i = np.zeros((n_dvr, n_spf))
    array_i[:, 0] = 1. / np.sqrt(n_dvr)
    for i in range(2 * dofs):
        basis[i].set_array(array_i)
    # Root state
    r_array = 1.0
    for i in range(2 * dofs):
        one_dim = np.zeros(n_spf)
        one_dim[0] = 1.0
        r_array = np.tensordot(r_array, one_dim, axes=0)
    root.set_array(r_array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    # single
    s_h = dvr.h_mat()
    for leaf in hamiltonian[:dofs]:
        h_list.append([(leaf, s_h)])
    # couple
    linear_ = partial(linear, c=c)
    dvr.set_v_func(linear_)
    l_h = dvr.v_mat()
    for i in range(dofs - 1):
        term = [(hamiltonian[i], l_h), (hamiltonian[i + 1], l_h)]
        h_list.append(term)

    # ML-MCTDH
    solver = MultiLayer(root, h_list)
    return solver
