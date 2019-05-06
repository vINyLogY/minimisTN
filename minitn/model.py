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
        ans[1:] = vec[:-1] * self.sqrt_sequence(dim)
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
        ans = LinearOperator((dim, dim), matvec=self.raising,
                             rmatvec=self.lowering)
        return ans

    @property
    def annihilation_operator(self):
        dim = self.dim
        ans = LinearOperator((dim, dim), matvec=self.lowering,
                             rmatvec=self.raising)
        return ans

    @property
    def number_operator(self):
        def matvec(x):
            return self.raising(self.lowering(x))

        dim = self.dim
        ans = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return ans

    @property
    def hami(self):
        def matvec(x):
            return self.raising(self.lowering(x))

        dim = self.dim
        ans = LinearOperator((dim, dim), matvec=matvec, rmatvec=matvec)
        return ans
