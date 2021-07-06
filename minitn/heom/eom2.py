#!/usr/bin/env python
# coding: utf-8
"""
Conversion:
    rho[n_0, ..., n_(k-1), ij]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np

from minitn.lib.tools import __, lazyproperty
from minitn.heom.noise import Correlation

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
        self.corr = corr
        assert sys_op.ndim == 2 
        assert sys_op.shape == sys_hamiltonian.shape
        self.n_states = sys_op.shape[0]
        self.op = sys_op
        self.h = sys_hamiltonian

    def gen_extended_rho(self, rho):
        """Get rho_n from rho with the conversion:
            rho[n_0, ..., n_(k-1), ij]

        Parameters
        ----------
        rho : np.ndarray
        """
        assert rho.ndim == 2 and rho.shape[0] == rho.shape[1]
        # rho_n[i, j, 0] = rho
        # rho_n[i, j, n] = 0
        ext = np.zeros((np.prod(self.n_dims),))
        ext[0] = 1
        rho_n = np.reshape(np.tensordot(ext, rho, axes=0), list(self.n_dims) + [-1])
        return rho_n

    def _comm(self, op):
        identity = np.identity(self.n_states)
        return np.kron(op, identity) - np.kron(identity, op)
    
    def _acomm(self, op):
        identity = np.identity(self.n_states)
        return np.kron(op, identity) + np.kron(identity, op)

    @property
    def sys_liouvillian(self):
        return self._comm(self.h) / (1.0j * self.hbar)

    @property
    def commutator(self):
        return self._comm(self.op)

    @property
    def acommutator(self):
        return self._acomm(self.op)

    def _upper(self, k):
        dim = self.n_dims[k]
        return np.eye(dim, k=-1)

    def _lower(self, k):
        dim = self.n_dims[k]
        return np.eye(dim, k=1)

    def _numberer(self, k):
        return np.diag(np.arange(self.n_dims[k]))

    def _diff_ij(self):
        array = self.sys_liouvillian - \
            self.corr.delta_coeff * np.matmul(self.commutator, self.commutator)
        return [[(self.k_max, array)]]

    def _diff_k(self, k):
        res = []
        gamma_k = self.corr.exp_coeff[k]
        s_k = self.corr.symm_coeff[k]
        a_k = self.corr.asymm_coeff[k]
        k_max = self.k_max
        
        # L_0
        ## gamma_k * np.einsum('k,...k...->...k...', nrange, rho_n)
        array0 = -gamma_k * self._numberer(k) 
        res.append([(k, array0)])

        # L_+
        array1_ij = (s_k * self.commutator + 1.0j * a_k * self.acommutator) / (1.0j * self.hbar)
        array1_k = np.matmul( self._upper(k), self._numberer(k),)
        res.append([(k_max, array1_ij), (k, array1_k)])

        # L_-
        array2_ij = self.commutator / (1.0j * self.hbar)
        array2_k = self._lower(k)
        res.append([(k_max, array2_ij), (k, array2_k)])

        return res

    def diff(self):
        """Get the derivative of rho_n at time t
        
        """
        derivative = self._diff_ij()

        for k in range(self.k_max):
            derivative.extend(self._diff_k(k))

        return derivative


# TODO: TD propagation

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
