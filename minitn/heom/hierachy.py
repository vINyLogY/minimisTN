#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.

Conversion:
    rho[n_0, ..., n_(k-1), i, j]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from itertools import product

from numpy import sqrt

from minitn.lib.backend import np

from minitn.lib.tools import __
from minitn.heom.corr import Correlation

DTYPE = np.complex128


class Hierachy(object):
    hbar = 1.0
    f_type = None

    def __init__(self, n_dims, sys_hamiltonian, sys_op, corr):
        """
        Parameters
        ----------
        n_dims : np.ndarray
            a vector representing the possible n
        sys_hamiltionian : np.ndarray
            H_s
        sys_op :
            X_s in in H_sb X_s (x) X_b 
        corr : Correlation
            Correlation caused by X_b
        """
        self.n_dims = n_dims
        self.k_max = len(n_dims)
        assert isinstance(corr, Correlation)
        assert self.k_max == corr.k_max
        self._i = len(n_dims)
        self._j = len(n_dims) + 1

        self.corr = corr
        assert sys_op.ndim == 2
        assert sys_op.shape == sys_hamiltonian.shape
        self.n_states = sys_op.shape[0]
        self.op = np.array(sys_op, dtype=DTYPE)
        self.h = np.array(sys_hamiltonian, dtype=DTYPE)

    def h_list(self, sys_i, sys_j, ph_indices):
        diff = self.diff()
        index_convension = list(ph_indices) + [sys_i, sys_j]
        h_list = []
        for term in diff:
            h_list.append([(index_convension[fst], snd) for fst, snd in term])

        return h_list

    def gen_extended_rho(self, rho):
        """Get rho_n from rho with the conversion:
            rho[n_0, ..., n_(k-1), i, j]

        Parameters
        ----------
        rho : np.ndarray
        """
        shape = list(rho.shape)
        assert len(shape) == 2 and shape[0] == shape[1]
        # Let: rho_n[0, i, j] = rho and rho_n[n, i, j] = 0
        ext = np.zeros((np.prod(self.n_dims),))
        ext[0] = 1
        rho_n = np.reshape(np.tensordot(ext, rho, axes=0), list(self.n_dims) + shape)
        return np.array(rho_n, dtype=DTYPE)

    def _raiser(self, k):
        """Acting on 0-th index"""
        dim = self.n_dims[k]
        sqrt_n = np.diag(np.sqrt(np.arange(dim, dtype=DTYPE)))
        return np.eye(dim, k=1, dtype=DTYPE) @ sqrt_n

    def _lower(self, k):
        """Acting on 0-th index"""
        dim = self.n_dims[k]
        sqrt_n = np.diag(np.sqrt(np.arange(dim, dtype=DTYPE)))
        return sqrt_n @ np.eye(dim, k=-1, dtype=DTYPE)

    def _numberer(self, k):
        return np.diag(np.arange(self.n_dims[k], dtype=DTYPE))

    def diff(self):
        """Get the derivative of rho_n at time t.
        
        Acting on 0-th index.
        """
        self.corr.print()
        i = self._i
        j = self._j
        derivative = [
            [(i, -1.0j * self.h)],
            [(j, 1.0j * self.h)],
        ]

        for k in range(self.k_max):
            ck = complex(self.corr.coeff[k])
            cck = complex(self.corr.conj_coeff[k])

            f_type = self.f_type
            if f_type == 1:
                fk = np.sqrt(ck + cck)
            elif f_type == 2:
                fk = np.sqrt(ck) + np.sqrt(cck)
            elif f_type == 3:
                fk = 0.5
            elif isinstance(f_type, float):
                fk = f_type
            else:
                fk = 1.0

            print(f"f_k (Type-{f_type}) = {fk}")
            dk = [
                [(k, self.corr.derivative[k] * self._numberer(k))],
                [(i, -1.0j * self.op), (k, ck / fk * self._raiser(k) + fk * self._lower(k))],
                [(j, 1.0j * self.op), (k, cck / fk * self._raiser(k) + fk * self._lower(k))],
            ]
            derivative.extend(dk)

        return derivative
