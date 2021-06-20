#!/usr/bin/env python
# coding: utf-8
r"""E-ph model and useful objects (operators, etc)
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator

from minitn.lib.tools import __


class Phonon(object):
    """Relevent operators for SHO with the basis diagonalizing the
    hamiltionian.
    """
    def __init__(self, dim, omega, mass=1, hbar=1, dense=True):
        r"""The Hamiltonian for a SHO is::

            H = \frac{p^2}{2 m} + \frac{m \omega^2}{2} x^2)

        """
        self.dim = dim
        self.omega = omega
        self.mass = mass
        self.hbar = hbar
        self.dense = dense
        return

    def check_vec(self, vec):
        if vec.shape != (self.dim,):
            raise TypeError("The shape of input vec is {}, but requires {}",
                            vec.shape, (self.dim,))
        else:
            return

    def raising(self, vec):
        self.check_vec(vec)
        ans = np.zeros_like(vec)
        ans[1:] = vec[:-1]
        ans *= np.array([np.sqrt(i) for i in range(self.dim)])
        return ans

    def lowering(self, vec):
        self.check_vec(vec)
        ans = np.zeros_like(vec)
        tmp = np.array([np.sqrt(i) for i in range(self.dim)]) * vec
        ans[:-1] = tmp[1:]
        return ans

    def operator(self, matvec, rmatvec=None):
        if rmatvec is None:
            rmatvec = matvec
        dim = self.dim
        op = LinearOperator((dim, dim), matvec=rmatvec, rmatvec=rmatvec)
        return op

    @property
    def creation_operator(self):
        if self.dense:
            ans = np.diag(np.sqrt(np.arange(1, self.dim)), -1)
        else:
            ans = self.operator(matvec=self.raising, rmatvec=self.lowering)
        return ans

    @property
    def annihilation_operator(self):
        if self.dense:
            ans = np.diag(np.sqrt(np.arange(1, self.dim)), 1)
        else:
            ans = self.operator(matvec=self.lowering, rmatvec=self.raising)
        return ans

    @property
    def coordinate_operator(self):
        coeff = np.sqrt(self.hbar / self.mass / self.omega / 2.)
        if self.dense:
            ans = coeff * (self.creation_operator + self.annihilation_operator)
        else:
            def matvec(x): return coeff * (self.raising(x) + self.lowering(x))
            ans = self.operator(matvec=matvec)
        return ans

    @property
    def momentum_operator(self):
        coeff = 1.0j * np.sqrt(self.hbar * self.mass * self.omega / 2.)
        if self.dense:
            ans = coeff * (self.creation_operator - self.annihilation_operator)
        else:
            def matvec(x): return coeff * (self.raising(x) - self.lowering(x))
            ans = self.operator(matvec=matvec)
        return ans

    @property
    def number_operator(self):
        if self.dense:
            ans = np.diag(np.arange(self.dim))
        else:
            def matvec(x): return self.raising(self.lowering(x))
            ans = self.operator(matvec=matvec)
        return ans

    @property
    def hamiltonian(self):
        coeff = self.hbar * self.omega
        if self.dense:
            ans = np.diag(coeff * (0.5 + np.arange(self.dim)))
        else:
            def matvec(x):
                return coeff * (self.raising(self.lowering(x)) + 0.5 * x)
            ans = self.operator(matvec=matvec)
        return ans
