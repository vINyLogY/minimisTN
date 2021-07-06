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
    ode_method = 'RK23'

    def __init__(self, wfn, h_list):
        """
        Parameters
        ----------
        wfn : WFN
        h_list : [[(Leaf, array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.  This is time independent part of Hamiltonian.
        """
        self.wfn = np.array(wfn, dtype='complex')
        self.h_list = h_list

        # For propogation purpose
        self.time = 0.0

        return

    def _next(self, tau):
        def diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            partial_product = Tensor.partial_product
            ans = np.zeros_like(self.wfn)
            for n, term in enumerate(self.h_list):
                tmp = np.reshape(y, self.wfn.shape)
                for i, array in term:
                    tmp = partial_product(tmp, i, array, 1)
                ans += tmp
            ans = np.reshape(ans, -1)
            return ans

        y0 = np.reshape(self.wfn, -1)
        solver = solve_ivp(diff, (self.time, self.time + tau), y0, method=self.ode_method)

        return np.reshape(solver.y[:, -1], self.wfn.shape)

    def propagator(self, steps=None, ode_inter=0.01, start=0):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
        method : {'RK23', 'RK45', ...}
        """
        for n in count():
            if steps is not None and n >= steps:
                break
            time = start + n * ode_inter
            self.time = time 
            yield (time, self.wfn)
            try:
                self.wfn = self._next(ode_inter)
            except:
                break
