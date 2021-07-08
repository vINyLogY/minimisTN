#!/usr/bin/env python
# coding: utf-8
"""Generating the derivative of the extended rho in SoP formalism.

Conversion:
    rho[n_0, ..., n_(k-1), i, j]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np

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
            X_s in in H_sb X_s x X_b 
        corr : Correlation
            Correlation case by X_b
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
        self.op = sys_op
        self.h = sys_hamiltonian

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

    def _upper(self, k):
        dim = self.n_dims[k]
        return np.eye(dim, k=-1)

    def _lower(self, k):
        dim = self.n_dims[k]
        return np.eye(dim, k=1)

    def _numberer(self, k, start=0):
        return np.diag(np.arange(start, start + self.n_dims[k]))

    def _sqrt_numberer(self, k, start=0):
        return np.diag(np.sqrt(np.arange(start, start + self.n_dims[k])))

    def _diff_ij(self):
        # delta = self.corr.delta_coeff

        ans = [
            [(self._i, -1.0j * np.transpose(self.h))],
            [(self._j, 1.0j * self.h)],
            # [(self._i, -delta * np.transpose(self.op @ self.op))],
            # [(self._i, np.sqrt(2.0) * delta * np.transpose(self.op)), 
            #  (self._j, np.sqrt(2.0) * delta * self.op)],
            # [(self._j, -delta * (self.op @ self.op))],
        ]
        
        return ans

    def _diff_k(self, k):
        gamma_k = self.corr.exp_coeff[k]
        c_k = self.corr.symm_coeff[k] + 1.0j * self.corr.asymm_coeff[k]
        upper = np.transpose(self._upper(k))
        lower = np.transpose(self._lower(k))
        numberer = self._numberer(k)

        ans = [
            [(k, - gamma_k * numberer)],
            [(self._i, -1.0j / self.hbar * np.transpose(self.op)), (k, lower)],
            [(self._j, 1.0j / self.hbar * self.op), (k, lower)],
            [(self._i, -1.0j / self.hbar * c_k * np.transpose(self.op)), (k, upper @ numberer)],
            [(self._j, 1.0j / self.hbar * np.conj(c_k) * self.op), (k, upper @ numberer)],
        ]

        return ans

    def diff(self):
        """Get the derivative of rho_n at time t
        
        """
        derivative = self._diff_ij()

        for k in range(self.k_max):
            derivative.extend(self._diff_k(k))

        return derivative



if __name__ == '__main__':
    from minitn.heom.noise import Drude
    from minitn.lib.units import Quantity

    # System
    e = Quantity(6500, 'cm-1').value_in_au
    v = Quantity(500, 'cm-1').value_in_au
    # Bath
    lambda_0 = Quantity(2000, 'cm-1').value_in_au # reorganization energy
    omega_0 = Quantity(2000, 'cm-1').value_in_au # vibrational frequency
    beta = Quantity(300, 'K').value_in_au # temperature
    # Superparameters
    max_terms = 5 # (terms used in the expansion of the correlation function)
    max_tier  = 10 # (number of possble values for each n_k in the extended rho)

    h = np.array([[0, v],
                  [v, e]])

    op = np.array([[0, 0],
                   [0, 1]])

    corr = Drude(lambda_0, omega_0, max_terms, beta)
    heom = Hierachy([max_tier] * max_terms, h, op, corr)
    phi = [1/np.sqrt(2), 1/np.sqrt(2)] 
    phi /= np.linalg.norm(phi)
    rho_0 = np.tensordot(phi, phi, axes=0)

    init_rho = heom.gen_extended_rho(rho_0)
    print(init_rho.shape)
    for n, term in enumerate(heom.diff()):
        print('- Term {}:'.format(n))
        for label, array in term: 
            print('Label: {}, shape: {}'.format(label, array.shape))
