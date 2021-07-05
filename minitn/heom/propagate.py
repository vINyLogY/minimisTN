#!/usr/bin/env python
# coding: utf-8
r"""ML-MCTDH Algorithms

References
----------
.. [1] arXiv:1603.03039v4
.. [2] My slides
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from functools import partial
from itertools import count, combinations_with_replacement
from contextlib import contextmanager

import numpy as np
from scipy import linalg, integrate, sparse
from scipy.integrate._ivp.ivp import solve_ivp

from minitn.lib.tools import __
from minitn.lib.numerical import DavidsonAlgorithm
from minitn.tensor import Tensor, Leaf


class MultiLayer(object):
    r"""A mini version of ML-MCTDH propagation method.
    """
    # Coefficient settings...
    hbar = 1.
    regular_err = 1.e-12
    svd_err = None
    svd_rank = None
    pinv = True
    max_ode_steps = 1000
    cmf_steps = 1
    ode_method = 'RK45'
    ode_in_real = False
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
        ps_method : string
            Method of projector-splitting.
            `s` for one-site method and `u` for two-site method.
        """
        for name, value in kwargs.items():
            if not hasattr(cls, name):
                raise AttributeError('{} has no attr \'{}\'!'
                                     .format(cls, name))
            setattr(cls, name, value)
        return

    def __init__(self, root, h_list, use_str_name=False):
        """
        Parameters
        ----------
        root : Tensor
        h_list : [[(Leaf, array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.  This is time independent part of Hamiltonian.
        f_list : [[(Leaf, float  ->  array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.  This is time dependent part of Hamiltonian.
        use_str_name : bool
            whether to use str to replace Leaf above. Default is False.
        """
        self.root = root
        self.h_list = h_list

        # For propogation purpose
        self.time = 0.0

        # Type check
        for term in h_list:
            for pair in term:
                if not isinstance(pair[0], Leaf) and not use_str_name:
                    raise TypeError('0-th ary in tuple must be of type Leaf!')
                if np.array(pair[1]).ndim != 2:
                    raise TypeError('1-th ary in tuple must be 2-D ndarray!')
        if use_str_name:
            leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
            for term in h_list:
                for _ in range(len(term)):
                    fst, snd = term.pop(0)
                    term.append((leaves_dict[str(fst)], snd))

        # Some pre-calculated data
        self.inv_density = {}    # {Tensor: ndarray}
        self.env_ = {}    # {(int, Tensor, int): ndarray}
        return

    def term_visitor(self):
        """Visit all terms in self.h_list.
        """
        visitor = self.root.visitor
        for tensor in visitor(axis=None):
            tensor.reset()
        for n, term in enumerate(self.h_list):
            for leaf, array in term:
                leaf.set_array(array)
            yield n
            for leaf, _ in term:
                leaf.reset()

    # @profile
    def _single_diff(self, tensor, n):
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

        # Env Hamiltonians
        tmp = tensor.array
        for i in range(tensor.order):
            env_ = self.env_[(n, tensor, i)]

            tmp = partial_product(tmp, i, env_)
        # For non-root nodes...
        if tensor.axis is not None:
            raise RuntimeError()
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

    # @profile
    def _form_env(self):
        self.env_ = {}
        network = self.root.visitor
        for n in self.term_visitor():
            for tensor in network(axis=None, leaf=False):
                for i in range(tensor.order):
                    env_ = tensor.partial_env(i, proper=True)
                    self.env_[(n, tensor, i)] = env_
        return self.env_

    def _solve_ode(self, diff, y0, ode_inter, reformer, updater):
        OdeSolver = getattr(integrate, self.ode_method)
        t0 = self.time
        t1 = t0 + ode_inter

        ode_solver = OdeSolver(diff, t0, y0, t1, vectorized=False)
        cmf_steps = self.cmf_steps
        for n in count(1):
            if ode_solver.status != 'running':
                logging.debug(__('* Propagation done.  Average CMF steps: {}',
                                n // cmf_steps))
                break
            if n % cmf_steps == 0:
                if n >= self.max_ode_steps:
                    msg = __('Reach ODE limit {}', n)
                    logging.warning(msg)
                    raise RuntimeWarning(msg)
                if reformer is not None:
                    reformer()
            ode_solver.step()
            updater(ode_solver.y)
        return

    def _split_prop(self, tensor, tau=0.01):
        def diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            origin = tensor.array
            tensor.set_array(np.reshape(y, tensor.shape))
            ans = np.zeros_like(y)
            for n in self.term_visitor():
                ans += np.reshape(self._single_diff(tensor, n), -1)
            tensor.set_array(origin)
            return ans

        if tensor.axis is None:
            self.root = tensor
        else:
            raise RuntimeError("Cannot propagate on Tensor {}:"
                               "Not a root node!".format(tensor))
        y0 = np.reshape(tensor.array, -1)
        #solver = solve_ivp(diff, (self.time, self.time + tau), y0, method=self.ode_method)
        #tensor.set_array(np.reshape(solver.y[:, -1], tensor.shape))

        def updater(a):
            tensor.set_array(np.reshape(a, tensor.shape))

        self._solve_ode(diff, y0, tau, None, updater)
        return tensor

    def remove_env(self, *args):
        for n, _ in enumerate(self.h_list):
            for tensor in args:
                for i, _, _ in tensor.linkages:
                    if (n, tensor, i) in self.env_:
                        del self.env_[(n, tensor, i)]
        return

    def move(self, t, i, op=None, unite_first=False):
        end, _ = t[i]
        self.remove_env(t, end)
        if unite_first:
            path = t.unite_split(i, operator=op, rank=self.svd_rank,
                                 err=self.svd_err)
        else:
            path = t.split_unite(i, operator=op)
        self.root = end
        self.remove_env(*path)
        return end

    @staticmethod
    def snd_order_step(step, ode_inter=0.01):
        half = 0.5 * ode_inter
        step(ode_inter=half, backward=False)
        step(ode_inter=half, backward=True)
        return 

    def split_step(self, ode_inter=0.01, backward=False):
        self._form_env()
        prop = partial(self._split_prop, tau=ode_inter)
        inv_prop = partial(self._split_prop, tau=(-ode_inter))
        linkages = list(self.root.decorated_linkage_visitor(leaf=False))
        move = self.move
        if backward:
            prop(self.root)
            for rev, (t1, _, t2, j) in reversed(linkages):
                if rev:
                    move(t2, j, op=inv_prop)
                    prop(t1)
                else:
                    move(t2, j)
        else:
            for rev, (t1, i, _, _) in linkages:
                if not rev:
                    move(t1, i)
                else:
                    prop(t1)
                    move(t1, i, op=inv_prop)
            prop(self.root)
        return

    def unite_step(self, ode_inter=0.01, backward=False):
        self._form_env()
        prop = partial(self._split_prop, tau=ode_inter)
        inv_prop = partial(self._split_prop, tau=(-ode_inter))
        linkages = list(self.root.decorated_linkage_visitor(leaf=False))
        move = self.move
        origin = self.root
        counter = {}
        for t in self.root.visitor(leaf=False):
            counter[t] = len(list(t.children(leaf=False)))
        if backward:
            n_origin = counter[origin]
            for rev, (t2, _, t1, i) in reversed(linkages):
                if rev:
                    if t2 is not origin or counter[t2] != n_origin:
                        inv_prop(t2)
                        counter[t2] -= 1
                    move(t1, i, op=prop, unite_first=True)
                else:
                    if counter[t2] > 0:
                        move(t1, i)
        else:
            for rev, (t1, i, t2, _) in linkages:
                if not rev:
                    if counter[t2] > 0:
                        move(t1, i)
                else:
                    move(t1, i, op=prop, unite_first=True)
                    counter[t2] -= 1
                    if t2 is not origin or counter[t2] > 0:
                        inv_prop(t2)
        assert not any(counter.values())
        return

    def propagator(self, steps=None, ode_inter=0.01, start=0):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
        method : {'RK23', 'RK45', ...}
        """
        identifier = self.ps_method.upper()
        if identifier.startswith('U'):
            step = self.unite_step
        elif identifier.startswith('S'):
            step = self.split_step
        else:
            raise ValueError("No PS method '{}'!".format(self.ps_method))
        if self.snd_order and not identifier.startswith('R'):
            step = partial(self.snd_order_step, step)
        root = self.root

        for n in count():
            if steps is not None and n >= steps:
                break
            time = start + n * ode_inter
            self.time = time 
            yield (time, root)
            try:
                step(ode_inter=ode_inter)
            except RuntimeWarning:
                break
