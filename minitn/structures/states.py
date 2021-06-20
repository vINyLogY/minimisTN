#!/usr/bin/env python
# coding: utf-8
r"""State data structure

State classes for TN type methods.

- Network:
    Most general type of tensor network.

- Tree:
    The type of tensor network used in ML-MCTDH.

- Chain:
    The type of tensor network used in (1D) DMRG.
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.lib.tools import __
from minitn.tensor import Tensor, Leaf

partial_product = Tensor.partial_product
partial_trace = Tensor.partial_trace


class Network(object):
    """
    Notes
    -----
    Methods in this class (and its subclass) should NOT have and side effect to
    the to the Tensor(s) in template of `self`. 
    """

    def __init__(self, template, wfn=None):
        """
        Parameters
        ----------
        template : Tensor
            Any of the Tensor in a network.
        wfn : {Tensor: ndarray}
            Tensor runs over the non-leaf vertices in the network connected to 
            self.root.
        """
        roots = [t for t in template.visitor(axis=None) if t.axis is None]
        if len(roots) != 1:
            raise NotImplementedError('Only support networks that have and'
                                      'only have one root node.')
        self.root = roots[0]
        self.wfn = wfn if wfn is not None else{}
        return

    def __getitem__(self, key):
        if key in self.vertices:
            try:
                value = self.wfn[key]
            except KeyError:
                raise KeyError('Tensor {} has no specific array.'.format(key))
            return value
        else:
            raise KeyError('No such Tensor {} in Network {}'.format(key, self))

    def __setitem__(self, key, value):
        if key in self.vertices:
            if self.wfn.get(key) is None:
                self.wfn[key] = value
            else:
                raise RuntimeError('Should not overwrite {}.wfn,'
                                   ' or try set {}.wfn derectly.'
                                   .format(self, self))
        else:
            raise KeyError('No such Tensor {} in Network {}'.format(key, self))


    def as_template(self):
        cls = type(self)
        void_wfn = {t: None for t in self.vertices}
        return cls(self.root, void_wfn)

    def copy(self):
        cls = type(self)
        copied_wfn = {t: np.array(a) for t, a in self.wfn.items()}
        return cls(self.root, copied_wfn)

    def deepcopy(self):
        if attributes is None:
            attributes = ['name', 'axis', 'normalized']
        cls = type(self)
        iso = {} # 'iso' means 'isomorphism'
        for t in self.vertices:
            TensorClass = type(t)
            s = TensorClass()
            for attr in attributes:
                setattr(s, attr, getattr(t, attr))
            iso[t] = s
        for t, i, s, j in self.edges:
            iso[t][i] = iso[s], j
        copied_wfn = {iso[t]: np.array(a) for t, a in self.wfn.items()}
        return cls(iso[self.root], copied_wfn)

    def split(self, tensor, axes=None, rank=None, err=None,
              normalized=False):
        pass

    def unite(self, tensor, axis, root=None, normalized=False):
        pass

    def split_unite(self, tensor, axis, operator=None, rank=None,
                    err=None, normalized=None):
        pass

    def unite_split(self, tensor, axis, operator=None, rank=None,
                    err=None, normalized=None):
        pass

    @property
    def vertices(self):
        return [node for node in self.root.visitor()]

    @property
    def edges(self):
        return [linkage for linkage in
               self.root.linkage_visitor(directed=False)]

    @property
    def directed_edges(self):
        return [linkage for linkage in
               self.root.linkage_visitor(directed=True)]

    def vectorize(self):
        """
        Return
        ------
        vec : (1D) ndarray
        """
        vec_list = []
        for t in self.vertices:
            vec_list.append(np.reshape(self.wfn[t], -1))
        ans = np.concatenate(vec_list, axis=None)
        return ans

    def tensorize(self, vec):
        """
        Parameters
        ----------
        vec : (1D) ndarray

        Return
        ------
        wfn : {Tensor: ndarray}
            Tensor runs over the non-leaf vertices in the network connected to 
            self.root.
        """
        new_net = self.as_template()
        start = 0
        for t in self.vertices:
            shape = t.shape
            end = start + np.prod(shape)
            new_net[t] = np.reshape(vec[start:end], shape)
            start = end
        return new_net


class Tree(Network):
    def __init__(self, template, wfn=None):
        super(Tree, self).__init__(template, wfn=wfn)
        return

    @property
    def leaves(self):
        return [t for t in self.vertices if isinstance(t, Leaf)]

    @property
    def subleaves(self):
        return [t[0][0] for t in self.leaves]
    
    @property
    def nonleaves(self):
        return [t for t in self.vertices if not isinstance(t, Leaf)]

    def partial_env(self, operator=None, conj_wfn=None):
        """
        Parameters
        ----------
        operator : Operator
        conj_wfn : with __getitem__

        Return
        ------
        env_dict : {(Tensor, int): ndarray}
        """
        env_dict = {}

        def _single_env(tensor, axis):
            # Check the cache
            if (tensor, axis) in env_dict:
                return env_dict[(tensor, axis)]
            # Main algorithm
            else:
                child, i = tensor[axis]
                if isinstance(child, Leaf):
                    if operator is None:
                        ans = None
                    else:
                        ans = operator[child]
                else:
                    wfn = self.wfn
                    child_env = [
                        (j, _single_env(grandchild, k))
                        for j, grandchild, k in child.children(axis=i)
                    ]
                    # Make use of the normalization condition
                    if (
                        conj_wfn is None and
                        axis == tensor.axis and
                        tensor.normalized and 
                        all(args[1] is None for args in child_env)
                    ):
                        ans = None
                    else:
                        temp = wfn[child]
                        for j, matrix in child_env:
                            temp = partial_product(temp, j, matrix)
                        conj = (wfn[child] if conj_wfn is None else
                                conj_wfn[child])
                        ans = partial_trace(temp, i, conj, i)
                env_dict[(tensor, axis)] = ans if ans is not None else 1.
                return

        for t in self.subleaves:
            for i in range(t.order):
                _single_env(t, i)
        env_dict[(self.root, None)] = _single_env(self.root, None)
        return env_dict


class Chain(Tree):
    def __init__(self, template, wfn=None):
        super(Chain, self).__init__(template, wfn=wfn)
        return
