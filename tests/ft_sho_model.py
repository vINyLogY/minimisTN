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
from minitn.lib.numerical import DavidsonAlgorithm
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR
from minitn.mctdh import MCTDH
from minitn.ml import MultiLayer


def square(x): return 0.5 * (x ** 2)


def linear(x, c=0.5): return c * x


def triangular(n):
    for i in range(2 * n):
        for j in range(i + 1):
            yield (i - j) * n + j


def test_2layers(lower= -5., upper=5., n_dvr=40, n_spf=5, dofs=2, c=0.5):
    # Create a graph
    root = Tensor(name='S')
    basis = []
    hamiltonian = []
    for i in range(dofs):
        bi = Tensor(name='B' + str(i), axis=0)
        basis.append(bi)
        hi = Leaf(name='P' + str(i))
        ai = Leaf(name='Q' + str(i))
        hamiltonian.extend((hi, ai))
        bi.link_to(0, root, i)
        bi.link_to(1, hi, 0)
        bi.link_to(2, ai, 0)

    # Generate initial state
    dvr = SineDVR(lower, upper, n_dvr)
    dvr.set_v_func(square)
    matvec = lambda x: np.dot(dvr.h_mat(), x)
    # SPFs
    array_i = np.ones((n_dvr,))
    array_i /= np.sqrt(n_dvr)
    da = DavidsonAlgorithm(matvec, init_vecs=[array_i], n_vals=n_spf)
    array_i = np.array(da.kernel(search_mode=True))
    a_i = []
    for n, i in enumerate(triangular(n_spf)):
        if n >= n_spf:
            break
        ii = i // n_spf
        jj = i % n_spf
        a_i.append(np.tensordot(array_i[ii], array_i[jj], axes=0))
    for i in range(dofs):
        basis[i].set_array(a_i)
    # Root state
    r_array = 1.0
    for i in range(dofs):
        one_dim = np.zeros((n_spf,))
        one_dim[0] = 1.0
        r_array = np.tensordot(r_array, one_dim, axes=0)
    root.set_array(r_array)

    # Check
    for t in root.visitor():
        t.check_completness(strict=True)

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
