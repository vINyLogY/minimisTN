#!/usr/bin/env python
# coding: utf-8

r"""Functions to discretized a spectral density (and other things about bath).

References
----------
.. [1] J. Chem. Phys. 154, 194104 (2021)
.. [2] J. Chem. Phys. 152, 204101 (2020)
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np

from minitn.lib.tools import __


class SpectrumFactory:
    @staticmethod
    def drude(lambda_, nu):
        def _drude(omega):
            res = 2.0 * lambda_ * (omega * nu) / (omega**2 + nu**2)
            return res
        return _drude


class Correlation(object):
    hbar = 1.0

    def __init__(self, k_max=None, beta=None):
        """
        k_max : int
            number of the terms in the basis functions
        beta : float
            Inverse temperature; 1 / (k_B T)
        """
        self.spectrum = None
        self.k_max = k_max
        self.beta = beta
        return

    def symm_coeff(self, k):
        's_k'
        assert 0 <= k < self.k_max
        return NotImplemented

    def asymm_coeff(self, k):
        'a_k'
        assert 0 <= k < self.k_max
        return NotImplemented

    def delta_coeff(self):
        's_delta'
        return NotImplemented
    
    def exp_coeff(self, k):
        'gamma_k'
        assert 0 <= k < self.k_max
        return NotImplemented


class Drude(Correlation):
    def __init__(self, lambda_, nu, k_max, beta):
        """
        Parameters
        ----------
        lambda_ : np.ndarray
        nu : np.ndarray
        """
        self.lambda_ = lambda_
        self.nu = nu
        super().__init__(k_max, beta)
        self.spectrum = SpectrumFactory.drude(lambda_, nu)
        return

    def exp_coeff(self, k):
        if k == 0:
            gamma = self.nu
        else:
            gamma = 2.0 * np.pi * k / (self.beta * self.hbar)
        return gamma

    def symm_coeff(self, k):
        v, l, bh = self.nu, self.lambda_, self.beta * self.hbar
        if k == 0:
            s = v * l / np.tan(bh * v / 2.0)
        else:
            s = 2.0 * self.spectrum(self.exp_coeff(k)) / bh
        return s

    def asymm_coeff(self, k):
        if k == 0:
            a = -self.nu * self.lambda_
        else:
            a = 0
        return a

    def delta_coeff(self):
        v, l, bh = self.nu, self.lambda_, self.beta * self.hbar
        d = np.sum([(self.symm_coeff(k) + 1.0j * self.asymm_coeff(k)) /
                     self.exp_coeff(k) 
                     for k in range(self.k_max)])
        return 2.0 * l / (bh * v) - d

