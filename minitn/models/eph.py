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
    def __init__(self, dim):
        self.dim = dim
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

    @property
    def creation_operator(self):
        dim = self.dim
        op = LinearOperator((dim, dim), matvec=self.raising,
                             rmatvec=self.lowering)
        return op

    @property
    def annihilation_operator(self):
        dim = self.dim
        op = LinearOperator((dim, dim), matvec=self.lowering,
                             rmatvec=self.raising)
        return op

    @property
    def coordinate_operator(self):
        def matvec(x):
            return (self.raising(x) + self.lowering(x)) / np.sqrt(2)

        dim = self.dim
        op = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return op

    @property
    def momentum_operator(self):
        def matvec(x):
            return -1.j * (self.raising(x) + self.lowering(x)) / np.sqrt(2)

        dim = self.dim
        op = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return op

    @property
    def number_operator(self):
        def matvec(x):
            return self.raising(self.lowering(x))

        dim = self.dim
        op = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return op

    @property
    def hamiltonian(self):
        def matvec(x):
            return self.raising(self.lowering(x)) + 0.5 * x

        dim = self.dim
        op = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return op
