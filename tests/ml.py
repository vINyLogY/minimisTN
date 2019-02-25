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
from minitn.ml import Multi_layer

@time_this
def test_2layers(lower, upper, n_dvr, n_spf, dofs, c):
    assert(n_spf < n_dvr)
    
    # Create a graph
    root = Tensor(name='S')
    basis = []
    hamiltonian = []
    for i in range(dofs):
        bi = Tensor(name='B' + str(i), axis=0)
        basis.append(bi)
        hi = Leaf(name='L' + str(i))
        hamiltonian.append(hi)
        bi.link_to(0, root, i)
        bi.link_to(1, hi, 0)

    # Generate initial state
    dvr = SineDVR(lower, upper, n_dvr)
    square = lambda x: 0.5 * (x ** 2)
    dvr.set_v_func(square)
    ## SPFs
    _, array_i = dvr.solve(n_state=n_spf)
    for i in range(dofs):
        basis[i].set_array(array_i)
    ## Root state
    r_array = 1.0
    for i in range(dofs):
        one_dim = np.zeros(n_spf)
        one_dim[0] = 1.0
        r_array = np.tensordot(r_array, one_dim, axes=0)
    root.set_array(r_array)

    # Set the hamiltonian
    h_list = []    # \sum_i x_i^2 + c^2 * \sum_i x_i * x_{i+1}
    ## single
    s_h = dvr.h_mat()
    for leaf in hamiltonian:
        h_list.append([(leaf, s_h)])
    ## couple
    linear = lambda x, c=c: c * x
    dvr.set_v_func(linear)
    l_h = dvr.v_mat()
    for i in range(dofs - 1):
        term = [(hamiltonian[i], l_h), (hamiltonian[i + 1], l_h)]
        h_list.append(term)

    # ML-MCTDH
    solver = Multi_layer(root, h_list)
    solver.eom()
    for tensor in root.visitor(leaf=False):
        msg = "tensor: {},\narray: {}".format(tensor, tensor.aux)
        print(msg)
        print(np.sum(tensor.local_inner_product()))
    
    print(root.global_norm())
    return

if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG + 1)
    test_2layers(lower=-5., upper=5., n_dvr=40, n_spf=5, dofs=2, c=0.5)

# EOF
