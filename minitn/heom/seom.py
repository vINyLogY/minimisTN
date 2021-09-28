#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended wfns in SoP formalism.

Conversion:
    c[n_0, ..., n_(k-1), i, k]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from itertools import product

import numpy as np
from scipy import linalg

from minitn.lib.tools import __
from minitn.heom.noise import Correlation

DTYPE = np.complex128


class Hierachy(object):
    hbar = 1.0

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
        self._q = len(n_dims)
        self._p = len(n_dims) + 1

        self.corr = corr
        assert sys_op.ndim == 2
        assert sys_op.shape == sys_hamiltonian.shape
        self.n_states = sys_op.shape[0]
        self.op = np.transpose(sys_op)
        self.h = np.transpose(sys_hamiltonian)
        return

    def creator(self, k):
        """Acting on 0-th index"""
        dim = self.n_dims[k]
        raiser = np.eye(dim, k=1)
        sqrt_n = np.diag(np.sqrt(np.arange(dim)))
        return raiser @ sqrt_n

    def annihilator(self, k):
        """Acting on 0-th index"""
        dim = self.n_dims[k]
        lower = np.eye(dim, k=-1)
        sqrt_n = np.diag(np.sqrt(np.arange(dim)))
        return sqrt_n @ lower

    def numberer(self, k):
        """Acting on 0-th index"""
        return np.diag(np.arange(self.n_dims[k]))

    def _diff_h(self):
        return [
            [(self._p, self.h)],
        ]

    def _diff_n(self):
        if self.corr.exp_coeff.ndim == 1:
            gamma = np.diag(self.corr.exp_coeff)
        ans = []
        for i, j in product(range(self.k_max), repeat=2):
            g = gamma[i, j]
            if not np.allclose(g, 0.0):
                term = [(i, 0.5j * self.hbar * g * self.numberer(i))]
                if i != j:
                    at = self.creator(i)
                    a = self.annihilator(j)
                    term.extend([(i, at), (j, a)])
                ans.append(term)
        return ans

    def _diff_k(self, k):
        c_k = self.corr.symm_coeff[k] + 1.0j * self.corr.asymm_coeff[k]
        at = self.creator(k)
        a = self.annihilator(k)

        return [
            [(self._p, self.op), (k, a)],
            [(self._p, c_k * self.op), (k, at)],
        ]

    def diff(self):
        """Get the derivative of rho_n at time t.
        
        Acting on 0-th index.
        """
        derivatives = self._diff_h() + self._diff_n()

        for k in range(self.k_max):
            derivatives.extend(self._diff_k(k))

        return derivatives


if __name__ == '__main__':
    from minitn.heom.noise import Drude
    from minitn.lib.units import Quantity

    # System
    e = Quantity(6500, 'cm-1').value_in_au
    v = Quantity(500, 'cm-1').value_in_au
    # Bath
    lambda_0 = Quantity(2000, 'cm-1').value_in_au  # reorganization energy
    omega_0 = Quantity(2000, 'cm-1').value_in_au  # vibrational frequency
    beta = Quantity(300, 'K').value_in_au  # temperature
    # Superparameters
    max_terms = 5  # (terms used in the expansion of the correlation function)
    max_tier = 10  # (number of possble values for each n_k in the extended rho)

    h = np.array([[0, v], [v, e]])

    op = np.array([[0, 0], [0, 1]])

    corr = Drude(lambda_0, omega_0, max_terms, beta)
    heom = Hierachy([max_tier] * max_terms, h, op, corr)
    phi = [1 / np.sqrt(2), 1 / np.sqrt(2)]
    phi /= np.linalg.norm(phi)
    rho_0 = np.tensordot(phi, phi, axes=0)

    init_rho = heom.gen_extended_rho(rho_0)
    print(init_rho.shape)
    for n, term in enumerate(heom.diff()):
        print('- Term {}:'.format(n))
        for label, array in term:
            print('Label: {}, shape: {}'.format(label, array.shape))
