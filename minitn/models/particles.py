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
    def __init__(self, dim, omega, mass=1, hbar=1):
        self.dim = dim
        self.omega = omega
        self.mass = mass
        self.hbar = hbar
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
        vec *= np.array([np.sqrt(i) for i in range(self.dim)])
        ans = np.zeros_like(vec)
        ans[:-1] = vec[1:]
        return ans

    def operator(self, matvec, rmatvec=None):
        if rmatvec is None:
            rmatvec = matvec
        dim = self.dim
        op = LinearOperator((dim, dim), matvec=rmatvec, rmatvec=rmatvec)
        return op

    @property
    def creation_operator(self):
        return self.operator(matvec=self.raising, rmatvec=self.lowering)

    @property
    def annihilation_operator(self):
        return self.operator(matvec=self.lowering, rmatvec=self.raising)

    @property
    def coordinate_operator(self):
        def matvec(x):
            coeff = np.sqrt(self.hbar / self.mass / self.omega / 2.)
            return coeff * (self.raising(x) + self.lowering(x))

        return self.operator(matvec=matvec)

    @property
    def momentum_operator(self):
        def matvec(x):
            coeff = 1.0j * np.sqrt(self.hbar * self.mass * self.omega / 2.)
            return coeff * (self.raising(x) - self.lowering(x))

        return self.operator(matvec=matvec)

    @property
    def number_operator(self):
        def matvec(x):
            return self.raising(self.lowering(x))

        return self.operator(matvec=matvec)

    @property
    def hamiltonian(self):
        def matvec(x):
            coeff = self.hbar * self.omega
            return coeff * (self.raising(self.lowering(x)) + 0.5 * x)

        return self.operator(matvec=matvec)
