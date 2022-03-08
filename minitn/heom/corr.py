#!/usr/bin/env python
# coding: utf-8
"""
Finite basis representation for bath correlation function 
using eigenbasis of ∂/∂t.

"""
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from typing import Optional, Tuple

from sympy import beta

from minitn.lib.backend import np, DTYPE
from minitn.lib.tools import __, lazyproperty


class Correlation(object):
    hbar = 1.0

    def __init__(self,
                 k_max=None,
                 beta=None,
                 coeff=None,
                 conj_coeff=None,
                 derivative=None):
        """
        k_max : int
            number of the terms in the basis functions
        beta : float
            Inverse temperature; 1 / (k_B T)
        """
        self.k_max = k_max
        self.beta = beta
        self.coeff = np.array(coeff,
                              dtype=DTYPE) if coeff is not None else None
        self.conj_coeff = np.array(
            conj_coeff, dtype=DTYPE) if conj_coeff is not None else None
        self.derivative = np.array(
            derivative, dtype=DTYPE) if derivative is not None else None
        return

    spectral_density = None

    def print(self):
        string = """Correlation coefficents:
            c: {};
            (c* = {};)
            gamma: {}.
        """.format(self.coeff, self.conj_coeff, self.derivative)
        print(string)


class Drude(Correlation):

    def __init__(self,
                 gamma,
                 lambda_,
                 k_max: int = 1,
                 beta: Optional[float] = None,
                 decompmethod=None):
        assert k_max >= 0

        if k_max == 0:
            c, g = [], [], []
        elif beta is None:
            c = [-1.0j * gamma * lambda_] + [0.0] * (k_max - 1)
            g = [-gamma] + [0.0] * (k_max - 1)
        else:
            if decompmethod is None:
                decompmethod = self.matsubara
            poles, residues = decompmethod(k_max - 1)

            def f_bose(x, poles, residues):
                return 1 / x + 0.5 + np.sum(2.0 * residues * x /
                                            (x**2 + poles**2))

            c = [
                -2.0j * lambda_ * gamma *
                f_bose(-1.0j * gamma * beta, poles, residues)
            ]
            g = [-gamma]

            for pole, res in zip(poles, residues):
                c.append(-4.0 * (pole / beta) * res * gamma * lambda_ /
                         (gamma**2 - (pole / beta)**2))
                g.append(-pole / beta)

        c = np.array(c, dtype=DTYPE)
        g = np.array(g, dtype=DTYPE)

        super().__init__(
            k_max=k_max,
            beta=beta,
            coeff=c,
            conj_coeff=np.conj(c),
            derivative=g,
        )
        self.gamma = gamma
        self.lambda_ = lambda_
        return

    @staticmethod
    def matsubara(n: int):
        poles = [2 * (i + 1) * np.pi for i in range(n)]
        residues = [1.0] * n
        return poles, residues

    @staticmethod
    def pade(n, method=-1):
        assert method in [-1]  # (N-1)/N method

        def tridiag_eigsh(diag, subdiag):
            mat = np.diag(subdiag, -1) + np.diag(diag) + np.diag(subdiag, 1)
            return np.sort(np.linalg.eigvalsh(mat))[::-1]

        if n > 0:
            diag_q = np.zeros(2 * n, dtype=float)
            subdiag_q = np.array([
                1.0 / np.sqrt((3 + 2 * i) * (5 + 2 * i))
                for i in range(2 * n - 1)
            ])
            poles = 2.0 / tridiag_eigsh(diag_q, subdiag_q)[:n]
            roots_q = np.power(poles, 2)

            diag_p = np.zeros(2 * n - 1, dtype=float)
            subdiag_p = np.array([
                1.0 / np.sqrt((5 + 2 * i) * (7 + 2 * i + 1))
                for i in range(2 * n - 2)
            ])
            roots_p = np.power(2.0 / tridiag_eigsh(diag_p, subdiag_p)[:n - 1],
                               2)

            residues = np.zeros(n, dtype=float)
            for i in range(n):
                res_i = 0.5 * n * (2 * n + 3) * (roots_p[i] - roots_q[i])
                for j in range(n):
                    if j != i:
                        res_i *= ((roots_p[j] - roots_q[i]) /
                                  (roots_q[j] - roots_q[i]))
                residues[i] = res_i

        return poles, residues

    @property
    def spectral_density(self):
        return SpectralDensityFactory.drude_lorentz(self.lambda_, self.gamma)


class Brownian(Correlation):

    def __init__(self, lambda_, omega, gamma, k_max=1, beta=None):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.omega = omega

        w1 = np.sqrt(omega**2 - gamma**2)  # underdamped: gamma < omega;
        # p = 4.0 * lambda_ * omega**2 * gamma

        if beta is None:
            coth1, coth2 = 1.0, 1.0
        else:
            coth1 = 1.0 / np.tanh(beta * (w1 - 1.0j * gamma) / 2.0)
            coth2 = 1.0 / np.tanh(beta * (w1 + 1.0j * gamma) / 2.0)
        coeff = [
            lambda_ * omega**2 / (2.0 * w1) * (coth1 + 1.0),
            lambda_ * omega**2 / (2.0 * w1) * (coth2 - 1.0),
        ]
        conj_coeff = [
            lambda_ * omega**2 / (2.0 * w1) * (np.conj(coth2) - 1.0),
            lambda_ * omega**2 / (2.0 * w1) * (np.conj(coth1) + 1.0),
        ]
        derivative = [
            -gamma - 1.0j * w1,
            -gamma + 1.0j * w1,
        ]

        if k_max > 1 and beta is not None:
            for k in range(1, k_max):
                vk = 2.0 * np.pi * k / beta
                dk = -2.0 * self.spectral_density(vk) / beta
                coeff.append(dk)
                conj_coeff.append(dk)
                derivative.append(-vk)

        super().__init__(
            k_max=k_max + 2,
            beta=beta,
            coeff=coeff,
            conj_coeff=conj_coeff,
            derivative=derivative,
        )
        return

    @property
    def spectral_density(self):
        return SpectralDensityFactory.brownian(self.lambda_, self.omega,
                                               self.gamma)


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
            gaussian = ((np.sqrt(np.pi) * lambda_g * omega) / (4. * omega_g) *
                        np.exp(-(omega / (2. * omega_g))**2))
            debye = ((lambda_d * omega * omega_d) / (2 *
                                                     (omega_d**2 + omega**2)))
            return gaussian + debye

        return _bimodal_spectral_density

    @staticmethod
    def brownian(lambda_, omega, gamma):

        def _brownian(w):
            return (4.0 * lambda_ * omega**2 * gamma * w /
                    ((w**2 - omega**2)**2 + (2.0 * gamma * w)**2))

        return _brownian
