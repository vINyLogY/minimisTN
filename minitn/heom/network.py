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
from scipy import linalg

from minitn.lib.tools import huffman_tree
from minitn.tensor import Tensor, Leaf
from minitn.models.network import autocomplete

DTYPE = np.complex128


def get_n_state(rho):
    shape = list(np.shape(rho))
    assert len(shape) == 2 and shape[0] == shape[1]
    return shape[0]


def simple_heom(init_rho, n_indices):
    """Get rho_n from rho with the conversion:
        rho[n_0, ..., n_(k-1), i, j]

    Parameters
    ----------
    rho : np.ndarray
    """
    n_state = get_n_state(init_rho)
    # Let: rho_n[0, :, :] = rho and rho_n[n, :, :] = 0
    ext = np.zeros((np.prod(n_indices),))
    ext[0] = 1.0
    new_shape = list(n_indices) + [n_state, n_state]
    rho_n = np.reshape(np.tensordot(ext, init_rho, axes=0), new_shape)

    root = Tensor(name='root', array=rho_n, axis=None)
    for k in range(len(new_shape)):  # +2: i and j
        l = Leaf(name=k)
        root[k] = (l, 0)

    return root


def simple_hseom(init_wfns, n_indices):
    n_state = get_n_state(init_wfns)
    # Let a[0, :, :] = wfn and a[n, :, :] = 0

    ext = np.zeros((np.prod(n_indices),))
    ext[0] = 1.0
    new_shape = list(n_indices) + [n_state, n_state]

    root_array = np.reshape(np.tensordot(ext, init_wfns, axes=0), new_shape)
    root = Tensor(name='root', array=root_array, axis=None)
    for k in range(len(new_shape)):  # +2: q and p
        l = Leaf(name=k)
        root[k] = (l, 0)

    return root


def get_ext_wfns(n_states, wfns, op, search_method='krylov'):
    wfns = np.array(wfns)
    n_states = np.shape(wfns)[1]
    space = np.transpose(np.array(wfns))
    vecs = np.transpose(np.array(wfns))
    assert np.shape(space)[1] <= n_states
    if search_method == 'krylov':
        while True:
            space = linalg.orth(vecs)
            if np.shape(space)[0] >= n_states:
                break
            vecs = list(op @ vecs)
            np.concatenate((space, vecs), axis=1)
        psi = space[:, :n_states]
        return np.transpose(psi)
    else:
        raise NotImplementedError


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

    # +2: i and j
    root[0] = (Leaf(name=max_terms), 0)
    root[1] = (Leaf(name=max_terms + 1), 0)

    for i in pb_index:
        assert rank <= i

    train = [root]
    for k in range(max_terms):
        if k < max_terms - 1:
            array = np.eye(rank, pb_index[k] * rank)
            array = np.reshape(array, (rank, -1, rank))
        else:
            array = np.eye(rank, pb_index[k])
        spf = Tensor(name=k, array=array, axis=0)
        l = Leaf(name=k)
        spf[0] = (train[-1], 2)
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
    graph, spf_root = huffman_tree(leaves, importances=importance, obj_new=new_spf, n_branch=3)

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
