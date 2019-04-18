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
from scipy import linalg, integrate, sparse

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
    svd_rank = None
    pinv = True
    max_ode_steps = 10000
    cmf_steps = 1
    ode_method = 'RK45'
    snd_order = False
    ps_method = 'split-unite'

    @classmethod
    def settings(cls, **kwargs):
        """
        Parameters
        ----------
        hbar : float
            Default = 1.
        regular_err : float
            Default = 1.e-12
        svd_err : float
            Error allowed for SVD; default is None.
        pinv : bool
            Whether to use `scipy.linalg.pinv2` for inversion.
            Default is True.
        max_ode_steps : int 
            Maximal steps allowed in one ODE solver; default = 1000.
        cmf_steps : int
            Upper bound for CMF steps; default = 1
        ode_method : {'RK45', 'RK23', ...}
            Name of `OdeSolver` in `scipy.intergate`.
        snd_order : boot
            Whether to use 2nd order method in projector splitting.
            Note that 2nd order method should be more accurate, but its
            complexity is :math:`2^d`, where `d` is the depth of tree.
            Default is False.
        """
        for name, value in kwargs.items():
            if not hasattr(cls, name):
                raise AttributeError('{} has no attr \'{}\'!'
                                     .format(cls, name))
            setattr(cls, name, value)
        return

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

    def expection(self, normalized=False):
        ans = 0.0
        for _ in self.term_visitor():
            ans += self.root.expection()
        if normalized:
            ans /= self.root.global_norm() ** 2
        return ans

    def _single_eom(self, tensor, n, cache=False):
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
                if cache:
                    self.env_[(n, tensor, i)] = env_
            tmp = partial_product(tmp, i, env_)
        # For non-root nodes...
        if tensor.axis is not None:
            # Inversion
            axis, inv = tensor.axis, self.inv_density[tensor]
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
        for tensor in visitor(axis=None):
            tensor.reset()
        for tensor in visitor(axis=None):
            axis = tensor.axis
            if axis is not None:
                density = tensor.partial_env(axis, proper=True)
                if self.pinv:
                    inv = linalg.pinv2(density)
                else:
                    inv = linalg.inv(density + self.regular_err *
                                           np.identity(tensor.shape[axis]))
                self.inv_density[tensor] = inv
        return self.inv_density

    def _form_env(self, root=None):
        self.env_ = {}
        if root is None:
            root = self.root
        network = root.visitor(axis=None, leaf=False)
        for n in self.term_visitor():
            for tensor in network:
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
        return -self.hbar if imaginary else 1.0j * self.hbar

    def _direct_step(self, ode_inter=0.01, imaginary=False):
        visitor = self.root.visitor
        self._form_env()
        self._form_inv_density()
        method = self.ode_method
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
            root = self.root

            def diff(t, y):
                """This function will not change the arrays in tensor network.
                """
                origin = root.vectorize()
                root.tensorize(y)
                self.eom(imaginary=imaginary)
                ans = root.vectorize(use_aux=True)
                root.tensorize(origin)
                return ans

            def reformer():
                self._form_env()
                self._form_inv_density()
                return

            def updater(y):
                root.tensorize(y)
                for t in root.visitor(leaf=False):
                    t.normalize()

            y0 = root.vectorize()
            self._solve_ode(diff, y0, ode_inter, reformer, updater)
        return

    def _solve_ode(self, diff, y0, ode_inter, reformer, updater):
        OdeSolver = getattr(integrate, self.ode_method)
        ode_solver = OdeSolver(diff, 0, y0, ode_inter, vectorized=False)
        cmf_steps = self.cmf_steps
        for n in count(1):
            if ode_solver.status != 'running':
                logging.debug(__('CMF steps: #{}, ', n // cmf_steps))
                break
            if n % cmf_steps == 0:
                if n >= self.max_ode_steps:
                    raise RuntimeWarning('Reach ODE limit {}'.format(n))
                reformer()
            ode_solver.step()
            updater(ode_solver.y)
        return

    def _split_prop(self, tensor, tau=0.01, imaginary=False, cache=False):
        def diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            origin = tensor.array
            tensor.set_array(np.reshape(y, tensor.shape))
            ans = np.zeros_like(y)
            for n in self.term_visitor(use_cache=True):
                ans += np.reshape(self._single_eom(tensor, n, cache=cache), -1)
            ans /= self.coefficient(imaginary=imaginary)
            tensor.set_array(origin)
            return np.reshape(ans, -1)

        def reformer(): return

        def updater(y):
            tensor.set_array(np.reshape(y, tensor.shape))
            tensor.normalize()

        y0 = np.reshape(tensor.array, -1)
        self._solve_ode(diff, y0, tau, reformer, updater)
        return tensor

    def remove_env(self, *args):
        for n, _ in enumerate(self.h_list):
            for tensor in args:
                for i, _, _ in tensor.linkages:
                    if (n, tensor, i) in self.env_:
                        del self.env_[(n, tensor, i)]
        return

    def move(self, t, i, op=None, unite_first=False, decorate=None):
        end, _ = t[i]
        if decorate is not None:
            decorate(t)
        self.remove_env(t, end)
        if unite_first:
            t.unite_split(i, operator=op, rank=self.svd_rank, err=self.svd_err)
        else:
            t.split_unite(i, operator=op)
        self.remove_env(t, end)
        if decorate is not None:
            decorate(end)
        return end

    def _split_step(self, ode_inter=0.01, imaginary=False,
                    _root=None, _axis=None):
        """Working projector-splitting method.  The time of the coefficient of
        a wfn matters most.
        """
        if _root is None:
            self._form_env()
            _root = self.root
        propagate = partial(self._split_prop, imaginary=imaginary)
        move = self.move

        def branch_prop(r, axis, tau, backward=False):
            unite_first = self.ps_method.startswith('u') and r is not self.root
            op1, op2, op3, op4, u1, u2 = (None, partial(propagate, tau=(-tau)),
                                          None, None, False, False)
            linkages = list(r.children(axis=axis, leaf=False))
            if unite_first:
                op2, op4, u2 = partial(propagate, tau=tau), op2, True
            if backward:
                op1, op2, op3, op4, u1, u2 = op2, op1, op4, op3, u2, u1
            if logging.root.isEnabledFor(logging.DEBUG):
                shape_dict = {}
                init = r.vectorize(shape_dict=shape_dict)
            for i, t, j in linkages:
                move(r, i, op1, u1, op3)
                self._split_step(ode_inter=tau, imaginary=imaginary,
                                 _root=t, _axis=j)
                move(t, j, op2, u2, op4)
                if logging.root.isEnabledFor(logging.DEBUG):
                    try:
                        r.tensorize(np.conj(init), use_aux=True,
                                    shape_dict=shape_dict)
                        ip = r.global_inner_product()
                    except:
                        ip = 'N/A'
                    logging.debug(__("r:{}({}); t:{}({}), <|>:{}",
                                     r, r.shape, t, t.shape, ip))
            return

        if self.snd_order:
            branch_prop(_root, _axis, 0.5 * ode_inter)
            propagate(_root, tau=ode_inter, cache=True)
            branch_prop(_root, _axis, 0.5 * ode_inter, backward=True)
        else:
            branch_prop(_root, _axis, ode_inter)
            propagate(_root, tau=ode_inter, cache=True)
        return

    def propagator(self, steps=None, ode_inter=0.01, split=False,
                   imaginary=False):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
        method : {'Newton', 'RK4', 'RK45', ...}
        """
        for n in count():
            if steps is not None and n > steps:
                break
            logging.info(__(
                "Propagating at t: {:.3f}, E: {:.8f}, |v|: {:.8f}",
                n * ode_inter,
                self.expection(normalized=True),
                self.root.global_norm()
            ))
            yield (n * ode_inter, self.root)
            step = self._split_step if split else self._direct_step
            try:
                step(ode_inter=ode_inter, imaginary=imaginary)
            except RuntimeWarning:
                raise StopIteration

    def autocorr(self, steps=None, ode_inter=0.01, split=False,
                 imaginary=False, fast=False):
        if not fast:
            _init = {}
            for t in self.root.visitor(leaf=False):
                _init[t] = t.array
        for time, r in self.propagator(steps=steps, ode_inter=ode_inter,
                                       split=split, imaginary=imaginary):
            for t in r.visitor(leaf=False):
                t.aux = t.array if fast else np.conj(_init[t])
            auto = r.global_inner_product()
            ans = (2. * time, auto) if fast else (time, auto)
            yield ans
            for t in r.visitor(leaf=False):
                t.aux = None

# EOF
