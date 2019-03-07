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
    for i in range(dofs):
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
    _, array_i = dvr.solve(n_state=n_spf)
    array_i = np.transpose(array_i)
    for i in range(dofs):
        basis[i].set_array(array_i)
    # Root state
    r_array = 1.0
    for i in range(dofs):
        one_dim = np.zeros(n_spf)
        one_dim[0] = 1.0
        r_array = np.tensordot(r_array, one_dim, axes=0)
    root.set_array(r_array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    # single
    s_h = dvr.h_mat()
    for leaf in hamiltonian:
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


def test_2layers(lower, upper, n_dvr, n_spf, dofs, c):
    assert(n_spf < n_dvr)

    # Create a graph
    root = Tensor(name='S')
    basis = []
    hamiltonian = []
    for i in range(dofs):
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
    _, array_i = dvr.solve(n_state=n_spf)
    array_i = np.transpose(array_i)
    for i in range(dofs):
        basis[i].set_array(array_i)
    # Root state
    r_array = 1.0
    for i in range(dofs):
        one_dim = np.zeros(n_spf)
        one_dim[0] = 1.0
        r_array = np.tensordot(r_array, one_dim, axes=0)
    root.set_array(r_array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    # single
    s_h = dvr.h_mat()
    for leaf in hamiltonian:
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


def test_mctdh(x0, x1, n_dvr, n_spf, dof, c):
    vf_list = [square] * dof
    conf_list = [[x0, x1, n_dvr]] * dof
    shape_list = [(n_dvr, n_spf)] * dof
    case = MCTDH(conf_list, shape_list)
    case.set_v_func(vf_list)
    linear_ = partial(linear, c=c)
    ex = []    # H_rst = cxy
    for i in range(dof - 1):
        ex.append([(i, linear_), (i + 1, linear_)])
    case.gen_h_terms(extra=ex, kinetic_only=False)
    case.init_state()
    return case
