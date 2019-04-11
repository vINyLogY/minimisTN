#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg

from minitn.lib.tools import __, time_this, figure
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


def test_2layers(lower=-5., upper=5., n_dvr=100, n_spf=5, dofs=4, c=0.5,
                 random_seed=None):
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
    np.random.seed(random_seed)
    for i in range(dofs):
        if random_seed is None:
            one_dim = np.zeros(n_spf)
            one_dim[0] = 1.0
        else:
            one_dim = np.random.random(size=(n_spf,))
            one_dim /= linalg.norm(one_dim)
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


def test_4layers(x0= -5., x1=5., n_1=5, n_2=5, n_3=5, n_4=40, c=0.5):
    """DOFs = 8, a full binary tree
    """
    graph = {
        0: [1, 2],    # 1
        1: [3, 4],    # 2
        2: [5, 6],
        3: [7, 8],    # 3
        4: [9, 10],
        5: [11, 12],
        6: [13, 14],
        7: [15],    # leaves
        8: [16],
        9: [17],
        10: [18],
        11: [19],
        12: [20],
        13: [21],
        14: [22],
    }
    dvr = SineDVR(x0, x1, n_4)
    dvr.set_v_func(square)
    _, array_i = dvr.solve(n_state=n_3)

    root = Tensor.generate(graph, 0)
    leaves = []
    for t in root.visitor():    # initial state
        if int(t.name) == 0:
            array = np.zeros((n_1, n_1))
            array[0, 0] = 1.
        elif 1 <= int(t.name) <= 2:
            array = np.zeros((n_1, n_2 ** 2))
            for n, v_i in zip(triangular(n_2), array):
                v_i[n] = 1.
            array = np.reshape(array, (n_1, n_2, n_2))
        elif 3 <= int(t.name) <= 6:
            array = np.zeros((n_2, n_3 ** 2))
            for n, v_i in zip(triangular(n_3), array):
                v_i[n] = 1.
            array = np.reshape(array, (n_2, n_3, n_3))
        elif 7 <= int(t.name) <= 14:
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
    linear_ = partial(linear, c=c)
    dvr.set_v_func(linear_)
    l_h = dvr.v_mat()
    for i in range(8 - 1):
        term = [(leaves[i], l_h), (leaves[i + 1], l_h)]
        h_list.append(term)

    for t in root.visitor():
        t.check_completness(strict=True)

    solver = MultiLayer(root, h_list)
    return solver


def test_mps_dmrg(x0=-5., x1=5., n_1=5, n_2=40, dofs=4, c=0.5):
    graph = {}
    for i in range(dofs):
        graph[str(i)] = ['B' + str(i)]
        graph['B' + str(i)] = ['L' + str(i)]
    for i in range(dofs - 1):
        graph[str(i)].append(str(i + 1))
    dvr = SineDVR(x0, x1, n_2)
    dvr.set_v_func(square)
    _, array_i = dvr.solve(n_state=n_1)

    root = Tensor.generate(graph, '0')
    leaves = []
    for t in root.visitor():
        if t.name == '0':
            array = np.zeros((n_1, n_1))
            array[0, 0] = 1.
        elif t.name == str(dofs - 1):
            array = np.eye(n_1, n_1)
            array = np.reshape(array, (n_1, n_1))        
        elif 'B' in t.name:
            array = array_i
        elif 'L' in t.name:
            array = None
            leaves.append(t)
        else:
            array = np.eye(n_1, n_1 ** 2)
            array = np.reshape(array, (n_1, n_1, n_1))
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
    for i in range(dofs - 1):
        term = [(leaves[i], l_h), (leaves[i + 1], l_h)]
        h_list.append(term)

    for t in root.visitor():
        t.check_completness(strict=True)

    solver = MultiLayer(root, h_list)
    return solver
