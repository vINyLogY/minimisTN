#!/usr/bin/env python
# coding: utf-8
"""
Finite basis representation for bath correlation function 
using eigenbasis of ∂/∂t.

"""
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from typing import Optional

from minitn.lib.backend import np, DTYPE
from minitn.lib.tools import __, lazyproperty


class Correlation(object):
    hbar = 1.0

    def __init__(self, k_max=None, beta=None, coeff=None, conj_coeff=None, derivative=None):
        """
        k_max : int
            number of the terms in the basis functions
        beta : float
            Inverse temperature; 1 / (k_B T)
        """
        self.spectrum = None
        self.k_max = k_max
        self.beta = beta
        self.coeff = np.array(coeff, dtype=DTYPE) if coeff is not None else None
        self.conj_coeff = np.array(conj_coeff, dtype=DTYPE) if conj_coeff is not None else None
        self.derivative = np.array(derivative, dtype=DTYPE) if derivative is not None else None
        return

    def print(self):
        string = """Correlation coefficents:
            c: {};
            (c* = {};)
            gamma: {}.
        """.format(self.coeff, self.conj_coeff, self.derivative)
        print(string)


class Drude(Correlation):

    def __init__(self, gamma, lambda_, k_max=1, beta=None):
        if k_max != 1:
            raise NotImplementedError
        if beta is None:
            s = 0.0
        else:
            # Naive high temperature
            s = 2 * lambda_ / beta
        a = -gamma * lambda_
        super().__init__(
            k_max=k_max,
            beta=beta,
            coeff=np.array([s + 1.0j * a]),
            conj_coeff=np.array([s - 1.0j * a]),
            derivative=np.array([-gamma]),
        )
        self.gamma = gamma
        self.lambda_ = lambda_
        return

    def spectral_density(self):
        return SpectralDensityFactory.drude_lorentz(self.lambda_, self.gamma)


class SpectralDensityFactory(object):

    @staticmethod
    def drude_lorentz(lambda_, gamma):

        def _drude(w):
            return (2.0 / np.pi) * (lambda_ * w * gamma) / (w**2 + gamma**2)

        return _drude

    @staticmethod
    def plain(eta, omega):

        def spec_func(w):
            if 0 < w < omega:
                return eta
            else:
                return 0.0

        return spec_func

    @staticmethod
    def bimodal_spectral_density(lambda_g, omega_g, lambda_d, omega_d):
        """C.f. J. Chem. Phys. 124, 034114 (2006). 
        
        Returns
        -------
            float  ->  float.
        """

        def _bimodal_spectral_density(omega):
            gaussian = ((np.sqrt(np.pi) * lambda_g * omega) / (4. * omega_g) * np.exp(-(omega / (2. * omega_g))**2))
            debye = ((lambda_d * omega * omega_d) / (2 * (omega_d**2 + omega**2)))
            return gaussian + debye

        return _bimodal_spectral_density
