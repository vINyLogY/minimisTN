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

from minitn.lib.tools import __, lazyproperty



class SpectrumFactory:
    @staticmethod
    def drude(lambda_, omega_0):
        def _drude(omega):
            res = 2.0 * lambda_ * (omega * omega_0) / (omega**2 + omega_0**2)
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

    symm_coeff = NotImplemented
    asymm_coeff = NotImplemented
    delta_coeff = NotImplemented
    exp_coeff = NotImplemented

    def print(self):
        string = """Correlation coefficents:
            S: {};
            (S_delta = {};)
            A: {};
            gamma: {}.
        """.format(self.symm_coeff,
                   self.delta_coeff,
                   self.asymm_coeff,
                   self.exp_coeff)
        print(string)


class Drude(Correlation):
    def __init__(self, lambda_, omega_0, k_max, beta):
        """
        Parameters
        ----------
        lambda_ : np.ndarray
        nu : np.ndarray
        """
        self.lambda_ = lambda_
        self.omega_0 = omega_0
        super().__init__(k_max, beta)
        self.spectrum = SpectrumFactory.drude(lambda_, omega_0)
        return


    @lazyproperty
    def exp_coeff(self):
        """Masturaba Frequencies"""
        def _gamma(k):
            if k == 0:
                gamma = self.omega_0
            else:
                gamma = 2.0 * np.pi * k / (self.beta * self.hbar)
            return gamma
        return np.array([_gamma(k) for k in range(self.k_max)])

    @lazyproperty
    def symm_coeff(self):
        v, l, bh = self.omega_0, self.lambda_, self.beta * self.hbar
        def _s(k):
            if k == 0:
                s = v * l / np.tan(bh * v / 2.0)
            else:
                s = 2.0 * self.spectrum(self.exp_coeff[k]) / bh
            return s
        return np.array([_s(k) for k in range(self.k_max)])

    @lazyproperty
    def asymm_coeff(self):
        def _a(k):
            if k == 0:
                a = -self.omega_0 * self.lambda_
            else:
                a = 0
            return a
        return np.array([_a(k) for k in range(self.k_max)])

    @lazyproperty
    def delta_coeff(self):
        t1 = 2.0 * self.lambda_ / (self.beta * self.hbar**2)
        t2 = np.sum([(self.symm_coeff + 1.0j * self.asymm_coeff) / (self.hbar * self.exp_coeff)])
        return t1 - t2

