#!/usr/bin/env python
# coding: utf-8
"""Template for Tensor network

Conversion:
    rho[n_0, ..., n_(k-1), i, j]
"""

from __future__ import absolute_import, division, print_function
from itertools import count

import logging
from builtins import filter, map, range, zip
from minitn.models.particles import Phonon

import numpy as np

from minitn.lib.tools import huffman_tree
from minitn.tensor import Tensor, Leaf
from minitn.models.network import autocomplete

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
    """Get rho_n from rho in a Tensor Train representation.

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
        assert rank <= i

    train = [root]
    for k in range(max_terms): # +2: i and j
        if k < max_terms - 1:
            array = np.eye(rank, pb_index[k] * rank)
            array = np.reshape(array, (rank, -1, rank))
        else:
            array = np.eye(rank, pb_index[k])
        spf = Tensor(name=k, array=array, axis=0)
        l = Leaf(name=k)
        spf[0] = (train[-1], train[-1].array.ndim - 1)
        spf[1] = (l, 0)
        train.append(spf)

    return train

def tensor_tree_template(init_rho, pb_index, rank=2):
    """Get rho_n from rho in a Tensor Tree representation.

    Parameters
    ----------
    rho : np.ndarray
    """
    n_state = get_n_state(init_rho)
    n_vec = np.zeros((rank,), dtype=DTYPE)
    n_vec[0] = 1.0
    root_array = np.tensordot(init_rho, n_vec, axes=0)
    max_terms = len(pb_index)

    for i in pb_index:
        assert rank <= i

    # generate leaves
    leaves = list(range(max_terms))
    class new_spf(object):
        counter = 0
        prefix = 'SPF'
        def __new__(cls):
            name = cls.prefix + str(cls.counter)
            cls.counter += 1
            return name

    importance = list(reversed(range(len(pb_index))))
    graph, spf_root = huffman_tree(leaves, importances=importance,
                                   obj_new=new_spf, n_branch=3)

    root = 'root'
    graph[root] = [spf_root, str(max_terms), str(max_terms + 1)]

    print(graph, root)

    root = Tensor.generate(graph, root)
    bond_dict = {}
    # Leaves
    l_range = list(pb_index) + [n_state] * 2
    for s, i, t, j in root.linkage_visitor():
        if isinstance(t, Leaf):
            bond_dict[(s, i, t, j)] = l_range[int(t.name)]
        else:
            bond_dict[(s, i, t, j)] = rank
    autocomplete(root, bond_dict)

    return root



