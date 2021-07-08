#!/usr/bin/env python
# coding: utf-8
"""Template for Tensor network

Conversion:
    rho[n_0, ..., n_(k-1), i, j]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np
from numpy.core.fromnumeric import shape

from minitn.lib.tools import __
from minitn.tensor import Tensor, Leaf

DTYPE=np.complex128


def get_n_state(rho):
    shape = list(np.shape(rho))
    assert len(shape) == 2 and shape[0] == shape[1]
    return shape[0]


def simple_heom(init_rho, pb_index):
    """Get rho_n from rho with the conversion:
        rho[n_0, ..., n_(k-1), i, j]

    Parameters
    ----------
    rho : np.ndarray
    """
    n_state = get_n_state(init_rho)
    # Let: rho_n[0, i, j] = rho and rho_n[n, i, j] = 0
    ext = np.zeros((np.prod(pb_index),))
    ext[0] = 1.0
    new_shape = list(pb_index) + [n_state, n_state]
    rho_n = np.reshape(np.tensordot(ext, init_rho, axes=0), new_shape)

    root = Tensor(name='root', array=rho_n, axis=None)
    for k in range(len(new_shape)): # +2: i and j
        l = Leaf(name=k)
        root[k] = (l, 0)

    return root


def tensor_train_template(init_rho, pb_index, rank=2):
    """Get rho_n from rho with the conversion:
        rho[i, j]

    Parameters
    ----------
    rho : np.ndarray
    """
    n_state = get_n_state(init_rho)
    n_vec = np.zeros((rank,), dtype=DTYPE)
    n_vec[0] = 1.0
    root_array = np.tensordot(init_rho, n_vec, axes=0)

    root = Tensor(name='root', array=root_array, axis=None)
    max_terms = len(pb_index)

    root[0] = (Leaf(name=max_terms), 0)
    root[1] = (Leaf(name=max_terms+1), 0)

    for i in pb_index:
        assert rank < i

    head = root
    for k in range(max_terms): # +2: i and j
        if k < max_terms - 1:
            array = np.eye(rank, pb_index[k] * rank)
            array = np.reshape(array, (rank, -1, rank))
        else:
            array = np.eye(rank, pb_index[k])
        spf = Tensor(name=k, array=array, axis=0)
        l = Leaf(name=k)
        spf[0] = (head, head.array.ndim - 1)
        spf[1] = (l, 0)
        head = spf

    return root




