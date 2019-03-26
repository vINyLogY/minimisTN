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
from itertools import count
from copy import copy
from collections import deque

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
    regular_err = 1.e-12
    svd_err = None
    pinv = True
    max_ode_steps = 1000
    cmf_steps = 1

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
        return

    @classmethod
    def settings(cls, **kwargs):
        for name, value in kwargs.items():
            setattr(cls, name, value)
        return

    def term_visitor(self, use_cache=False):
        """Visit all terms in self.h_list.
        """
        visitor = self.root.visitor
        for tensor in visitor(axis=None):
            tensor.reset()
        for n, term in enumerate(self.h_list):
            for tensor in visitor(axis=None, leaf=False):
                tensor.reset()
                if use_cache:
                    for i, t, j in tensor.linkages:
                        if (n, t, j) in self.env_:
                            tensor.load_cache(i, self.env_[(n, t, j)])
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
            if (n, tensor, i) not in self.env_:
                self.env_[(n, tensor, i)] = partial_env(i, proper=True)
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
        return tmp

    def _form_inv_density(self):
        self.inv_density = {}
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
                        MultiLayer.regular_err *
                        np.identity(tensor.shape[axis])
                    )
        return self.inv_density

    def _form_env(self):
        self.env_ = {}
        visitor = self.root.visitor
        for n in self.term_visitor():
            for tensor in visitor(leaf=False):
                for i in range(tensor.order):
                    env_ = tensor.partial_env(i, proper=True)
                    self.env_[(n, tensor, i)] = env_
        return self.env_

    def eom(self, check=False, imaginary=False):
        r"""Write the derivative of each Tensor in tensor.aux.

                   .
            g ::= <t|t> = 0

        Parameters
        ----------
        check : bool
            True to check the linkage completness.
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

    def _direct_step(self, ode_inter=0.01, method='RK45', imaginary=False):
        def _vec_diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            origin = root.vectorize()
            root.tensorize(y)
            self.eom(imaginary=imaginary)
            ans = root.vectorize(use_aux=True)
            root.tensorize(origin)
            return ans

        visitor = self.root.visitor
        if method == 'Newton':
            self.eom(imaginary=imaginary)
            for t in visitor(leaf=False):
                y0 = t.array
                dy = ode_inter * t.aux
                t.set_array(y0 + dy)
                t.aux = None
        elif method == 'RK4':
            k = [{}, {}, {}, {}]  # save [y0, k1, k2, k3]
            eom = partial(self.eom, imaginary=imaginary)
            eom()    # for k1
            for t in visitor(leaf=False):
                y0 = t.array
                k1 = ode_inter * t.aux
                t.set_array(y0 + k1 / 2)
                k[0][t] = y0
                k[1][t] = k1
            eom()    # for k2
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k2 = ode_inter * t.aux
                t.set_array(y0 + k2 / 2)
                k[2][t] = k2
            eom()    # for k3
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k3 = ode_inter * t.aux
                t.set_array(y0 + k3)
                k[3][t] = k3
            eom()    # for k4
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k4 = ode_inter * t.aux
                t.set_array(
                    k[0][t] +
                    (k[1][t] + 2. * k[2][t] + 2. * k[3][t] + k4) / 6.
                )
                t.aux = None
        else:
            OdeSolver = getattr(integrate, method)
            cmf_steps = MultiLayer.cmf_steps
            root = self.root
            y0 = root.vectorize()
            ode_solver = OdeSolver(_vec_diff, 0, y0, ode_inter,
                                   vectorized=False)
            for n in count(1):
                if ode_solver.status != 'running':
                    logging.debug(__('CMF: #{}, ', n // cmf_steps))
                    break
                if n % cmf_steps == 0:
                    if n >= MultiLayer.max_ode_steps:
                        raise RuntimeWarning('Reach ODE limit {}'.format(n))
                    self._form_env()
                    self._form_inv_density()
                ode_solver.step()
                root.tensorize(ode_solver.y)
        return

    def _split_prop(self, tensor, tau=0.01, imaginary=False, method='RK45'):
        def _vec_diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            origin = tensor.array
            tensor.set_array(np.reshape(y, tensor.shape))
            ans = np.zeros_like(y)
            for n in self.term_visitor(use_cache=True):
                ans += np.reshape(self._single_eom(tensor, n), -1)
            ans /= self.coefficient(imaginary=imaginary)
            tensor.set_array(origin)
            return np.reshape(ans, -1)

        OdeSolver = getattr(integrate, method)
        cmf_steps = MultiLayer.cmf_steps
        y0 = np.reshape(tensor.array, -1)
        ode_solver = OdeSolver(_vec_diff, 0, y0, tau, vectorized=False)
        for n in count(1):
            if ode_solver.status != 'running':
                logging.debug(__('CMF@{}: #{}, ', tensor, n // cmf_steps))
                break
            if n % cmf_steps == 0:
                if n >= MultiLayer.max_ode_steps:
                    raise RuntimeWarning('Reach ODE limit {}'.format(n))
                self._form_env()
            ode_solver.step()
            tensor.set_array(np.reshape(ode_solver.y, tensor.shape))
            tensor.normalize()
        return tensor

    def remove_env(self, tensor_list):
        try:
            it = iter(tensor_list)
        except TypeError:
            it = [tensor_list]
        n_term = len(self.h_list)
        for tensor in it:
            n_order = tensor.order
            for n in range(n_term):
                for i in range(n_order):
                    if (n, tensor, i) in self.env_:
                        del self.env_[(n, tensor, i)]
        return

    def _split_step(self, ode_inter=0.01, method='RK45',
                    imaginary=False, _root=None, _axis=None):
        """
        FIXME: Unstable.
        """
        err = MultiLayer.svd_err
        if _root is None:
            self._form_env()
            _root = self.root
        propagate = partial(self._split_prop,
                            method=method, imaginary=imaginary)

        def _branch_prop(r, axis, tau, backward=False):
            _inv_prop = partial(propagate, tau=(-tau))
            for i, t, j in r.children(axis=axis, leaf=False):
                op = _inv_prop if backward else None
                self.remove_env((r, t))
                mid = r.split_unite(i, operator=op, err=err)
                self.remove_env(mid)
                self._split_step(ode_inter=tau, method=method,
                                 imaginary=imaginary, _root=t, _axis=j)
                op = None if backward else _inv_prop
                self.remove_env((r, t))
                mid = t.split_unite(j, operator=op, err=err)
                self.remove_env(mid)
            return

        _branch_prop(_root, _axis, 0.5 * ode_inter, backward=False)
        self.remove_env(_root)
        propagate(_root, tau=ode_inter)
        _branch_prop(_root, _axis, 0.5 * ode_inter, backward=True)
        return

    def propagator(self, steps=None, ode_inter=0.01,
                   method='RK45', split=False, imaginary=False):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
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
            try:
                if split:
                    self._form_env()
                    self._split_step(ode_inter=ode_inter, method=method,
                                     imaginary=imaginary)
                else:
                    self._form_env()
                    self._form_inv_density()
                    self._direct_step(ode_inter=ode_inter, method=method,
                                      imaginary=imaginary)
            except RuntimeWarning:
                raise StopIteration
            if imaginary:
                self.root.normalize()
            _i += 1

    def autocorr(self, steps=None, ode_inter=0.01,
                 method='RK45', split=False, fast=True, imaginary=False):
        if not fast:
            _init = {}
            for t in self.root.visitor(leaf=False):
                _init[t] = t.array
        for time, r in self.propagator(steps=steps, ode_inter=ode_inter,
                                       method=method, split=split,
                                       imaginary=imaginary):
            for t in r.visitor(leaf=False):
                t.aux = t.array if fast else np.conj(_init[t])
            auto = r.global_inner_product()
            ans = (2. * time, auto) if fast else (time, auto)
            yield ans

# EOF
