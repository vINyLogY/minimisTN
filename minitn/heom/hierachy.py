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

from minitn.lib.backend import np

from minitn.lib.tools import __
from minitn.heom.corr import Correlation
from minitn.bases.dvr import SineDVR

DTYPE = np.complex128


class Hierachy(object):
    hbar = 1.0
    scale = 1.0

    def __init__(self, n_dims, sys_hamiltonian, sys_op, corr, basis=None):
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
        if basis is None:
            basis = [None] * self.k_max
        self.basis = basis
        assert sys_op.ndim == 2
        assert sys_op.shape == sys_hamiltonian.shape
        self.n_states = sys_op.shape[0]
        self.op = np.array(sys_op, dtype=DTYPE)
        self.h = np.array(sys_hamiltonian, dtype=DTYPE)

    def h_list(self, sys_i, sys_j, ph_indices):
        index_convension = list(ph_indices) + [sys_i, sys_j]
        h_list = []
        for term in self._diff():
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
        ext = np.zeros((np.prod(self.n_dims), ))
        ext[0] = 1
        rho_n = np.reshape(np.tensordot(ext, rho, axes=0),
                           list(self.n_dims) + shape)
        return np.array(rho_n, dtype=DTYPE)

    def _raiser(self, k):
        """Acting on 0-th index"""
        if self.basis[k] is None:
            dim = self.n_dims[k]
            sqrt_n = np.diag(np.sqrt(np.arange(dim, dtype=DTYPE)))
            ans = np.eye(dim, k=1, dtype=DTYPE) @ sqrt_n
        else:
            q = self.basis[k].q_mat() / np.sqrt(2)
            dq = self.basis[k].dq_mat() / np.sqrt(2)
            ans = np.transpose(np.array(q - dq, dtype=DTYPE))
            print(
                f'raiser-{k}:',
                np.diagonal(
                    self.basis[k].eig_mat().T @ ans @ self.basis[k].eig_mat(),
                    offset=1,
                )[:10])
        return ans

    def _lower(self, k):
        """Acting on 0-th index"""
        if self.basis[k] is None:
            dim = self.n_dims[k]
            sqrt_n = np.diag(np.sqrt(np.arange(dim, dtype=DTYPE)))
            ans = sqrt_n @ np.eye(dim, k=-1, dtype=DTYPE)
        else:
            q = self.basis[k].q_mat() / np.sqrt(2)
            dq = self.basis[k].dq_mat() / np.sqrt(2)
            ans = np.transpose(np.array(q + dq, dtype=DTYPE))
            print(
                f'lower-{k}:',
                np.diagonal(
                    self.basis[k].eig_mat().T @ ans @ self.basis[k].eig_mat(),
                    offset=-1,
                )[:10])
        return ans

    def _numberer(self, k):
        """Acting on 0-th index"""
        if self.basis[k] is None:
            ans = np.diag(np.arange(self.n_dims[k], dtype=DTYPE))
        else:
            n = len(self.basis[k].grid_points)
            q2 = self.basis[k].q_mat()**2
            dq2 = self.basis[k].dq2_mat()
            ans = np.transpose(0.5 * (q2 - dq2 - np.identity(n)))
            print(
                f'numberer-{k}:',
                np.diagonal(
                    self.basis[k].eig_mat().T @ ans @ self.basis[k].eig_mat(),
                    offset=0,
                )[:10])
        return ans

    def _diff(self):
        """Get the derivative of rho_n at time t.
        
        Acting on 0-th index.
        """
        #self.corr.print()
        i = self._i
        j = self._j
        derivative = [
            [(i, -1.0j * self.h)],
            [(j, 1.0j * self.h)],
        ]

        for k in range(self.k_max):
            ck = complex(self.corr.coeff[k])
            cck = complex(self.corr.conj_coeff[k])
            if self.basis[k] is None:
                fk = self.scale
            else:
                fk = np.sqrt(np.real((ck + cck) / 2.0))
            print(f"f_{k}: {fk}")

            dk = [
                [(k, self.corr.derivative[k] * self._numberer(k))],
                [(i, -1.0j * self.op),
                 (k, ck / fk * self._raiser(k) + fk * self._lower(k))],
                [(j, 1.0j * self.op),
                 (k, cck / fk * self._raiser(k) + fk * self._lower(k))],
            ]
            derivative.extend(dk)

        return derivative
