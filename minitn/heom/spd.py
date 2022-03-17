#!/usr/bin/env python
# coding: utf-8
"""
Decomposition of the bath and BE distribution
"""
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from math import gamma
from typing import Literal, Optional, Tuple

from numpy.typing import ArrayLike
from sympy import residue

from minitn.lib.backend import np, DTYPE

PI = np.pi


class BoseEinstein(object):

    def __init__(self, beta: Optional[float]) -> None:
        if beta is not None:
            assert beta >= 0

        self.beta = beta

        return

    def function(self, w):
        beta = self.beta
        if beta is None:
            return 1.0
        else:
            return 1.0 / (1.0 - np.exp(-beta * w))

    def residues(self, n: int, decomp_method='pade'):
        if decomp_method.lower() == 'pade':
            method = self.pade
        elif decomp_method.lower() == 'matsubara':
            method = self.matsubara
        else:
            raise NotImplementedError

        b = self.beta
        if b is None:
            return []
        else:
            residues = method(n)
            _u = [(r / b, -1.0j * z / b) for r, z in residues]
            _n = [(r / b, 1.0j * z / b) for r, z in residues]
            return (_u + _n)

    @staticmethod
    def matsubara(n: int):
        return [(1.0, 2.0 * PI * (i + 1)) for i in range(n)]

    @staticmethod
    def pade(n: int, _type: Literal[-1, 0, 1] = -1):
        assert _type in [-1]  # (N-1)/N method
        assert n >= 0

        def eigvalsh(subdiag):
            mat = np.diag(subdiag, -1) + np.diag(subdiag, 1)
            return sorted(np.linalg.eigvalsh(mat), reverse=True)

        if n == 0:
            residues = []
        else:
            subdiag_q = np.array([
                1.0 / np.sqrt((3 + 2 * i) * (5 + 2 * i))
                for i in range(2 * n - 1)
            ])
            zetas = 2.0 / eigvalsh(subdiag_q)[:n]
            roots_q = np.power(zetas, 2)

            subdiag_p = np.array([
                1.0 / np.sqrt((5 + 2 * i) * (7 + 2 * i + 1))
                for i in range(2 * n - 2)
            ])
            roots_p = np.power(2.0 / eigvalsh(subdiag_p)[:n - 1], 2)

            residues = []
            for i in range(n):
                res_i = 0.5 * n * (2 * n + 3)
                if i < n - 1:
                    res_i *= (roots_p[i] - roots_q[i]) / (roots_q[n - 1] -
                                                          roots_q[i])
                for j in range(n - 1):
                    if j != i:
                        res_i *= ((roots_p[j] - roots_q[i]) /
                                  (roots_q[j] - roots_q[i]))
                residues.append(res_i)

        return [(r, z) for r, z in zip(residues, zetas)]


class Bath(object):

    def __init__(self):
        pass

    def spectral_density(self, w: ArrayLike) -> ArrayLike:
        return NotImplemented

    def residues(self) -> list[Tuple[complex, complex]]:
        return NotImplemented


class Drude(Bath):

    def __init__(self, lambda_, gamma) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma

        return

    def spectral_density(self, w):
        l = self.lambda_
        g = self.gamma
        return (2.0 / PI) * l * g * w / (w**2 + g**2)

    def residues(self):
        l = self.lambda_
        g = self.gamma
        return [(l * g / PI, -1.0j * g), (l * g / PI, 1.0j * g)]


class Brownian(object):

    def __init__(self, lambda_, gamma, omega_0) -> None:
        self.lambda_ = lambda_
        self.gamma = gamma
        self.omega_0 = omega_0

        return

    def spectral_density(self, w):
        l = self.lambda_
        g = self.gamma
        w0 = self.omega_0

        return ((4.0 / PI) * l * g * (w0**2 + gamma**2) * w /
                ((w - w0)**2 + g**2) / ((w + w0)**2 + g**2))

    def residues(self):
        l = self.lambda_
        g = self.gamma
        w0 = self.omega_0

        return [
            (l * g * (w0**2 + gamma**2) / (PI * w0), -1.0j * (g + 1.0j * w0)),
            (l * g * (w0**2 + gamma**2) / (PI * w0), -1.0j * (g - 1.0j * w0)),
            (l * g * (w0**2 + gamma**2) / (PI * w0), 1.0j * (g + 1.0j * w0)),
            (l * g * (w0**2 + gamma**2) / (PI * w0), 1.0j * (g - 1.0j * w0)),
        ]
