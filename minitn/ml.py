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
from scipy import linalg, integrate

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
    err = 1.e-12
    pinv = True

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
        # Type check and initialize leaf._array with None
        for term in h_list:
            for leaf, array in term:
                if not isinstance(leaf, Leaf):
                    raise TypeError('0-th ary in tuple must be of type Leaf!')
                if np.array(array).ndim != 2:
                    raise TypeError('1-th ary in tuple must be 2-D ndarray!')
                leaf.reset()
        self.inv_density = {}    # {Tensor: ndarray}
        self.env_ = {}    # {(int, Tensor, int): ndarray}
        self._init = {}    # {Tensor: ndarray}
        return

    @classmethod
    def settings(cls, hbar=1., err=1.e-6, pinv=False):
        cls.hbar = hbar
        cls.err = err
        cls.pinv = pinv
        return

    def term_visitor(self):
        visitor = self.root.visitor
        for n, term in enumerate(self.h_list):
            for t in visitor():
                t.reset()
            for leaf, array in term:
                leaf.set_array(array)
            yield (n, visitor)

    def matrix_element(self):
        ans = 0.0
        for _ in self.term_visitor():
            ans += self.root.matrix_element()
        return ans

    def expection(self):
        ans = 0.0
        for _ in self.term_visitor():
            ans += self.root.expection()
        return ans

    def eom(self, check=False, cmf=False):
        r"""Write the derivative of each Tensor in tensor.aux.

                   .
            g ::= <t|t> = 0

        Parameters
        ----------
        check : bool
            True to check the linkage completness.
        cmf : bool
            Whether to re-calculate self.inv_density and self.env_ 
        """
        visitor = self.root.visitor
        for t in visitor():
            t.reset()
            t.aux = None

        if check:
            for tensor in visitor():
                tensor.check_completness(strict=True)

        # All partial densities
        if not cmf:
            for tensor in visitor():
                axis = tensor.axis
                if axis is not None:
                    density = tensor.partial_env(axis, proper=True)
                    if Multi_layer.pinv:
                        self.inv_density[tensor] = linalg.pinv2(density)
                    else:
                        self.inv_density[tensor] = linalg.inv(
                            density + 
                            Multi_layer.err * np.identity(tensor.shape[axis])
                        )

        # Term by term...
        partial_product = Tensor.partial_product
        partial_trace = Tensor.partial_trace
        for n, network in self.term_visitor():
            for tensor in network(leaf=False):
                partial_env = tensor.partial_env

                # Env Hamiltonians
                tmp = tensor.array
                for i in range(tensor.order):
                    if not cmf:
                        env_ = partial_env(i, proper=True)
                        self.env_[(n, tensor, i)] = env_
                    else:
                        env_ = self.env_[(n, tensor, i)]
                    tmp = partial_product(tmp, i, env_)

                # For non-root nodes...
                axis = tensor.axis
                if axis is not None:
                    # Inversion
                    inv = self.inv_density[tensor]
                    tmp = partial_product(tmp, axis, inv)
                    # Projection
                    tmp_1 = np.array(tmp)
                    array = tensor.array
                    conj_array = np.conj(array)
                    tmp = partial_trace(tmp, axis, conj_array, axis)
                    tmp = partial_product(array, axis, tmp, j=1)
                    tmp = (tmp_1 - tmp)

                prev = tensor.aux
                tensor.aux = tmp if prev is None else prev + tmp

        for tensor in visitor(leaf=False):
            tensor.aux /= 1.0j * Multi_layer.hbar
        return

    def propagator(
        self, end=None, ode_inter=0.01, cmf_step=None, method='RK45'
    ):
        """Propagator generator

        Parameters
        ----------
        end : int
        ode_inter : float
        cmf_step : {int, None}
        method : {'Newton', 'RK4', 'RK45', ...}
        """
        _i = 0
        while True:
            if end is not None and _i >= end:
                raise StopIteration()

            logging.info(__(
                "t: {:.3f}, E: {:.8f}, |v|^2: {:.8f}",
                _i * ode_inter, self.expection(),
                (self.root.global_norm()) **2
            ))
            yield (_i * ode_inter, self.root)
            cmf = (cmf_step is not None and _i % cmf_step != 0)
            _i += 1
            visitor = self.root.visitor
            if method == 'Newton':
                self.eom(cmf=cmf)
                for t in visitor(leaf=False):
                    y0 = t.array
                    dy = ode_inter * t.aux
                    t.set_array(y0 + dy)
                    t.aux = None
            elif method == 'RK4':
                k = [{}, {}, {}, {}]    # save [y0, k1, k2, k3]
                self.eom(cmf=cmf)    # for k1
                for t in visitor(leaf=False):
                    y0 = t.array
                    k1 = ode_inter * t.aux
                    t.set_array(y0 + k1 / 2)
                    k[0][t] = y0
                    k[1][t] = k1
                self.eom(cmf=cmf)    # for k2
                for t in visitor(leaf=False):
                    y0 = k[0][t]
                    k2 = ode_inter * t.aux
                    t.set_array(y0 + k2 / 2)
                    k[2][t] = k2
                self.eom(cmf=cmf)    # for k3
                for t in visitor(leaf=False):
                    y0 = k[0][t]
                    k3 = ode_inter * t.aux
                    t.set_array(y0 + k3)
                    k[3][t] = k3
                self.eom(cmf=cmf)    # for k4
                for t in visitor(leaf=False):
                    y0 = k[0][t]
                    k4 = ode_inter * t.aux
                    t.set_array(
                        k[0][t] +
                        (k[1][t] + 2. * k[2][t] + 2. * k[3][t] + k4) / 6.
                    )
                    t.aux = None
            else:    # use scipy.integrate API
                root = self.root

                def _vec_diff(t, y):
                    root.tensorize(y)
                    self.eom(cmf=cmf)
                    ans = root.vectorize(use_aux=True)
                    return ans

                y0 = root.vectorize()
                ode_solver = integrate.solve_ivp(
                    _vec_diff, (0., ode_inter), y0, method=method
                )
                y1 = np.transpose(ode_solver.y)[-1]
                root.tensorize(y1)

    def autocorr(
        self, end=None, ode_inter=0.01, cmf_step=None, method='RK45', fast=True
    ):
        if not fast:
            self._init = {}
            for t in self.root.visitor(leaf=False):
                self._init[t] = t.array
        for time, r in self.propagator(
            end=end, ode_inter=ode_inter, cmf_step=cmf_step, method=method
        ):
            for t in r.visitor(leaf=False):
                t.aux = t.array if fast else np.conj(self._init[t])
            auto = r.global_inner_product()
            ans = (2. * time, auto) if fast else (time, auto)
            yield ans

# EOF
