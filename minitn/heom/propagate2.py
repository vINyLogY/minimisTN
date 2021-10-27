#!/usr/bin/env python
# coding: utf-8
r"""ML-MCTDH Algorithms with projector-splitting method only.
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from functools import partial
from itertools import count, combinations_with_replacement
from contextlib import contextmanager

from minitn.lib.backend import np
from scipy import linalg, integrate, sparse
from scipy.integrate import solve_ivp

from minitn.lib.tools import __
from minitn.lib.numerical import DavidsonAlgorithm
from minitn.tensor import Tensor, Leaf


class ProjectorSplitting(object):
    r"""A mini version of ML-MCTDH propagation method.
    """
    # Coefficients and Superparameters...
    hbar = 1.
    svd_err = None
    svd_rank = None
    ode_method = 'RK45'
    snd_order = False
    atol = 1.0e-11
    rtol = 1.0e-8
    ps_method = 'split-unite'

    def __init__(self, root: Tensor, h_list):
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
        """
        self.root = root
        self.h_list = h_list

        # For propogation purpose
        self.time = 0.0

        # Some pre-calculated data
        self.env_ = {}  # {(int, Tensor, int): ndarray}
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

    def _form_env(self):
        self.env_ = {}
        network = self.root.visitor
        for n in self.term_visitor():
            for tensor in network(axis=None, leaf=False):
                for i in range(tensor.order):
                    env_ = tensor.partial_env(i, proper=True)
                    self.env_[(n, tensor, i)] = env_
        return self.env_

    def _single_diff(self, tensor, n):
        """C.f. `Multi-Configuration Time Dependent Hartree Theory: a Tensor
        Network Perspective`, p38.

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
            try:
                env_ = self.env_[(n, tensor, i)]
            except KeyError:
                env_ = tensor.partial_env(i, proper=True)
            tmp = partial_product(tmp, i, env_)

        # For non-root nodes: PS not allowed
        if tensor.axis is not None:
            raise RuntimeError()
        return tmp

    def _single_prop(self, tensor, tau=0.01):
        if tensor.axis is None:
            self.root = tensor
        else:
            raise RuntimeError("Cannot propagate on Tensor {}:" "Not a root node!".format(tensor))

        def diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            tensor.set_array(np.reshape(y, tensor.shape))
            ans = np.zeros_like(y)
            for n in self.term_visitor():
                ans += np.reshape(self._single_diff(tensor, n), -1)
            return ans

        y0 = np.reshape(tensor.array, -1)
        solver = solve_ivp(diff, (self.time, self.time + tau),
                           y0,
                           method=self.ode_method,
                           atol=self.atol,
                           rtol=self.rtol)
        tensor.set_array(np.reshape(solver.y[:, -1], tensor.shape))

        return tensor

    def remove_env(self, *args):
        for n, _ in enumerate(self.h_list):
            for tensor in args:
                for i, _, _ in tensor.linkages:
                    if (n, tensor, i) in self.env_:
                        del self.env_[(n, tensor, i)]
        return

    def move(self, t, i, op=None, unite_first=False):
        # if __debug__:
        #     print("[DEBUG] Moving {} along axis {}".format(t, i))
        end, _ = t[i]
        self.remove_env(t, end)
        if unite_first:
            path = t.unite_split(i, operator=op, rank=self.svd_rank, err=self.svd_err)
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

    def one_site_step(self, ode_inter=0.01, backward=False):
        self._form_env()
        prop = partial(self._single_prop, tau=ode_inter)
        inv_prop = partial(self._single_prop, tau=(-ode_inter))
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

    def two_site_step(self, ode_inter=0.01, backward=False):
        self._form_env()
        prop = partial(self._single_prop, tau=ode_inter)
        inv_prop = partial(self._single_prop, tau=(-ode_inter))
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
            step = self.two_site_step
        elif identifier.startswith('S'):
            step = self.one_site_step
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
            step(ode_inter=ode_inter)


class PnC(object):
    """A Propagation-Compression method

    Inspired by TD-DMRG (?)
    """
    pass
