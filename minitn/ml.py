#!/usr/bin/env python
# coding: utf-8
r"""Tensor data structure

Interface to numpy.ndarray.

References
----------
.. [1] arXiv:1603.03039v4
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg

from minitn.lib.tools import __
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR

class Multi_layer(object):
    r"""Structure of the wavefunction/state::

          ... ... ... 
        n_0|   |   |n_p-1
          2_0 ... 2_p-1
             \ | /
          m_0 \|/ m_p-1
               1
               |
              ...

    and corresponding Hamiltonian:
    * Option 1::

          n_0/   \n_p-1
           h_0   h_p-1
        n_0/ \   / \n_p-1
               + r
               |
           -- ... --

    Note that the order of contraction is essential.
    """
    # Coefficient settings...
    hbar = 1.
    err = 1.e-6

    def __init__(self, root, h_list):
        """
        Parameters
        ----------
        root : Tensor
        h_list : [[(Leaf, array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.
        """
        self.root = root
        self.h_list = h_list
        for term in h_list:
            # Type check and initialize leaf._array with None
            for leaf, array in term:
                if not isinstance(leaf, Leaf):
                    raise TypeError('0-th ary in tuple must be of type Leaf!')
                if np.array(array).ndim != 2:
                    raise TypeError('1-th ary in tuple must be 2-D ndarray!')
                leaf.reset()
        return

    @classmethod
    def settings(cls, hbar=1., err=1.e-6):
        cls.hbar = hbar
        cls.err = err
        return

    def eom(self):
        """Write the derivative of each Tensor in tensor.aux.

        Parameters
        ----------
        err : float
            Noise used in inversion.
        hbar : float
            Default is 1.0
        """
        visitor = self.root.visitor
        # All partial densities (and checks)...
        density = {}
        for tensor in visitor():
            tensor.check_completness(strict=True)
            tensor.aux = None
            axis = tensor.axis
            if axis is None:
                if tensor is self.root or isinstance(tensor, Leaf):
                    continue
                elif isinstance(tensor, Leaf):
                    tensor.reset()
                    continue
                else:
                    raise RuntimeError(
                        'In ML there cannot be more than one root nodes!'
                    )
            else:
                density[tensor] = tensor.partial_env(axis, proper=True)
        # Term by term...
        for term in self.h_list:
            for leaf, array in term:
                leaf.set_array(array)
            for tensor in visitor(leaf=False):
                partial_env = tensor.partial_env
                partial_product = Tensor.partial_product
                # Env Hamiltonians
                tmp = tensor.array
                for i in range(tensor.order):
                    env_ = partial_env(i, proper=True)
                    tmp = partial_product(tmp, i, env_)
                axis = tensor.axis
                if axis is not None:
                    m = tensor.shape[axis]
                    # Trick step: inversion
                    inv = linalg.inv(
                        density[tensor] + np.identity(m) * Multi_layer.err
                    )
                    tmp = partial_product(tmp, axis, inv)
                    # Projection
                    tmp_1 = np.array(tmp)
                    array = tensor.array
                    conj_array = np.conj(array)
                    tmp = Tensor.partial_trace(tmp, axis, conj_array, axis)
                    tmp = partial_product(array, axis, tmp, j=1)
                    tmp = (tmp_1 - tmp)
                out = tensor.aux
                tensor.aux = tmp if out is None else out + tmp
        for tensor in visitor(leaf=False):
            tensor.aux /= 1.0j * Multi_layer.hbar
        return

    def propagator(self, diff, ode_inter=0.01):
        pass


# EOF
