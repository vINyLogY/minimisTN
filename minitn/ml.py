#!/usr/bin/env python
# coding: utf-8
r"""ML-MCTDH Algorithms

References
----------
.. [1] arXiv:1603.03039v4
.. [2] My slides
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial
from copy import copy

import numpy as np
from scipy import linalg, integrate

from minitn.lib.tools import __
from minitn.tensor import Tensor, Leaf
from minitn.dvr import SineDVR


class MultiLayer(object):
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
    svd_err = None
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

        # Some cached data
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
            for t in visitor(axis=None):
                t.reset()
            for leaf, array in term:
                leaf.set_array(array)
            yield n
            for leaf, _ in term:
                leaf.reset()

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

    def _single_eom(self, tensor, n):
        """C.f. `Multi-Configuration Time Dependent Hartree Theory: a Tensor
        Network Perspective`, p38. This method does not contain the `i hbar`
        coefficient.

        Parameters
        ----------
        tensor : Tensor
            Must in a graph with all nodes' array set, including the leaves.
        n : int
            No. of Hamiltonian term.

        Return:
        -------
        array : ndarray
            With the same shape with tensor.shape.
        """
        partial_product = Tensor.partial_product
        partial_trace = Tensor.partial_trace
        partial_env = tensor.partial_env

        # Env Hamiltonians
        tmp = tensor.array
        for i in range(tensor.order):
            try:
                env_ = self.env_[(n, tensor, i)]
            except KeyError:
                env_ = partial_env(i, proper=True)
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
        return tmp

    def _form_inv_density(self):
        visitor = self.root.visitor
        for tensor in visitor():
            tensor.reset()
        for tensor in visitor():
            axis = tensor.axis
            if axis is not None:
                density = tensor.partial_env(axis, proper=True)
                if MultiLayer.pinv:
                    self.inv_density[tensor] = linalg.pinv2(density)
                else:
                    self.inv_density[tensor] = linalg.inv(
                        density +
                        MultiLayer.err * np.identity(tensor.shape[axis])
                    )
        return self.inv_density

    def _form_env(self):
        visitor = self.root.visitor
        for n in self.term_visitor():
            for tensor in visitor(leaf=False):
                for i in range(tensor.order):
                    env_ = tensor.partial_env(i, proper=True)
                    self.env_[(n, tensor, i)] = env_
        return self.env_

    def eom(self, check=False, cmf=False, imaginary=False):
        r"""Write the derivative of each Tensor in tensor.aux.

                   .
            g ::= <t|t> = 0

        Parameters
        ----------
        check : bool
            True to check the linkage completness.
        cmf : bool
            Whether to re-calculate self.inv_density and self.env_ 
        imaginary : bool
            Whether to treat t as it.
        """
        visitor = self.root.visitor
        if check:
            for tensor in visitor():
                tensor.check_completness(strict=True)
        # Clean
        for t in visitor():
            t.aux = None
        # All partial densities
        if not cmf:
            self._form_inv_density()
            self._form_env()
        # Term by term...
        for n in self.term_visitor():
            for tensor in visitor(leaf=False):
                tmp = self._single_eom(tensor, n)
                prev = tensor.aux
                tensor.aux = tmp if prev is None else prev + tmp
        # Times coefficient
        for tensor in visitor(leaf=False):
            tensor.aux /= self.coefficient(imaginary=imaginary)
        return

    def coefficient(self, imaginary=False):
        return -MultiLayer.hbar if imaginary else 1.0j * MultiLayer.hbar

    def _direct_step(self, ode_inter=0.01, cmf=False, method='RK45',
                     imaginary=False):
        visitor = self.root.visitor
        if method == 'Newton':
            self.eom(cmf=cmf, imaginary=imaginary)
            for t in visitor(leaf=False):
                y0 = t.array
                dy = ode_inter * t.aux
                t.set_array(y0 + dy)
                t.aux = None
        elif method == 'RK4':
            k = [{}, {}, {}, {}]    # save [y0, k1, k2, k3]
            self.eom(cmf=cmf, imaginary=imaginary)    # for k1
            for t in visitor(leaf=False):
                y0 = t.array
                k1 = ode_inter * t.aux
                t.set_array(y0 + k1 / 2)
                k[0][t] = y0
                k[1][t] = k1
            self.eom(cmf=cmf, imaginary=imaginary)    # for k2
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k2 = ode_inter * t.aux
                t.set_array(y0 + k2 / 2)
                k[2][t] = k2
            self.eom(cmf=cmf, imaginary=imaginary)    # for k3
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k3 = ode_inter * t.aux
                t.set_array(y0 + k3)
                k[3][t] = k3
            self.eom(cmf=cmf, imaginary=imaginary)    # for k4
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
                self.eom(cmf=cmf, imaginary=imaginary)
                ans = root.vectorize(use_aux=True)
                return ans

            y0 = root.vectorize()
            ode_solver = integrate.solve_ivp(
                _vec_diff, (0., ode_inter), y0, method=method
            )
            y1 = np.transpose(ode_solver.y)[-1]
            root.tensorize(y1)
        return

    def _split_prop(self, tensor, tau=0.01, imaginary=False, method='RK45'):
        shape = tensor.shape

        def _d(t, y):
            tensor.set_array(np.reshape(y, shape))
            d = np.zeros_like(y)
            for n in self.term_visitor():
                d += np.reshape(self._single_eom(tensor, n), -1)
            d /= self.coefficient(imaginary=imaginary)
            return np.reshape(d, -1)

        y0 = np.reshape(tensor.array, -1)
        ode_solver = integrate.solve_ivp(_d, (0., tau), y0, method=method)
        y1 = np.transpose(ode_solver.y)[-1]
        tensor.set_array(np.reshape(y1, shape))
        tensor.normalize()
        return

    def _split_step(self, ode_inter=0.01, cmf=False, method='RK45', err=None,
                    imaginary=False, split_first=True):
        """
        TODO:
        * Propagating from top (close to leaves) to bottom (root)
          (try IDDFS?)
        """
        if err is None:
            err = MultiLayer.svd_err
        propagate = partial(self._split_prop, imaginary=imaginary,
                            method=method)
        # if not cmf:
        #     self._form_env()
        if split_first:
            # Now the visitor is DFS and has 2 directions
            visiting_list = list(self.root.linkage_visitor(leaf=False,
                                                           back=True))
            for t1, i, t2, j in visiting_list:
                order1 = len(list(t1.children(axis=None, leaf=False)))
                inter1 = 0.5 * ode_inter / order1
                order2 = len(list(t2.children(axis=None, leaf=False)))
                inter2 = 0.5 * ode_inter / order2
                if __debug__:
                    link1, link2 = copy(t1._access), copy(t2._access)
                # t1 prop inter1
                propagate(t1, tau=inter1)
                # t1 split mid
                mid, _ = t1.split(i, indice=(0, i), err=err, child=t1)
                # mid prop -inter1
                propagate(mid, tau=-inter1)
                # t2 unite mid
                t2.unite(j, root=t2)
                # t2 prop inter2
                propagate(t2, tau=inter2)
                assert(t1._access == link1)
                assert(t2._access == link2)
        else:    # FIXME
            visiting_list = list(
                self.root.linkage_visitor(leaf=False, back=True)
            )
            for t1, i, t2, j in visiting_list:
                if __debug__:
                    link1, link2 = copy(t1._access), copy(t2._access)
                order1 = len(list(t1.children(axis=None, leaf=False)))
                inter1 = ode_inter / order1
                axes = list(range(t2.order - 1))
                # unite t1 and t2 as mid
                mid = t2.unite(j)
                # propagate mid with inter1
                propagate(mid, tau=inter1)
                # split mid to t2' and t1', where t2' is new root.
                mid.split(axis=axes, indice=(j, i), root=t2, child=t1)
                # propagate t2' with -inter1
                propagate(t2, tau=-inter1)
                assert(t1._access == link1)
                assert(t2._access == link2)
        return

    def propagator(
        self, steps=None, ode_inter=0.01, cmf_step=None, method='RK45',
        split=False, imaginary=False
    ):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
        cmf_step : {int, None}
        method : {'Newton', 'RK4', 'RK45', ...}
        """
        _i = 0
        while steps is None or _i < steps:
            logging.info(__(
                "Propagating at t: {:.3f}, E: {:.8f}, |v|: {:.8f}",
                _i * ode_inter, self.expection(),
                (self.root.global_norm()) ** 2
            ))
            yield (_i * ode_inter, self.root)
            cmf = (cmf_step is not None and _i % cmf_step != 0)
            if split:
                self._split_step(
                    ode_inter=ode_inter, cmf=cmf, method=method,
                    imaginary=imaginary
                )
            else:
                self._direct_step(
                    ode_inter=ode_inter, cmf=cmf, method=method,
                    imaginary=imaginary
                )
            if imaginary:
                for t in self.root.visitor(leaf=False):
                    t.normalize()
            _i += 1

    def autocorr(
        self, steps=None, ode_inter=0.01, cmf_step=None, method='RK45',
        split=False, fast=True, imaginary=False
    ):
        if not fast:
            self._init = {}
            for t in self.root.visitor(leaf=False):
                self._init[t] = t.array
        for time, r in self.propagator(
            steps=steps, ode_inter=ode_inter, cmf_step=cmf_step, method=method,
            split=split, imaginary=imaginary
        ):
            for t in r.visitor(leaf=False):
                t.aux = t.array if fast else np.conj(self._init[t])
            auto = r.global_inner_product()
            ans = (2. * time, auto) if fast else (time, auto)
            yield ans

# EOF
