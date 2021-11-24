#!/usr/bin/env python
# coding: utf-8
r"""Functions to discretized a spectral density (and other things about bath).

References
----------
.. [1] J. Chem. Phys. 124, 034114 (2006)
       https://doi.org/10.1063/1.2161178
.. [2] Phy. Rev. B. 92, 155126 (2015)
       http://dx.doi.org/10.1103/PhysRevB.92.155126
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from typing import Optional

from minitn.lib.backend import np, DTYPE
from scipy.integrate import quad

from minitn.lib.tools import __
from minitn.heom.corr import Correlation


def linear_discretization(spec_func, stop, num, start=0.0):
    """A simple linear method to discretize a spectral density.

    Parameters
    ----------
    spec_func : float  ->  float
        Offen denoted as J(w).
    start : float, optional
        Start point of the spectrum, defalut is 0.0.
    stop : float
        End point of the spectrum.
    num : int
        Number of modes to be given.

    Returns
    -------
    ans : [(float, float)] 
        `ans[i][0]` is the omega of one mode and `ans[i][1]` is the 
        corrospoding coupling in second quantization for all `i` in 
        `range(0, num)`.
    """

    def direct_quad(a, b):
        density = quad(spec_func, a, b)[0]
        omega = quad(lambda x: x * spec_func(x), a, b)[0] / density
        coupling = np.sqrt(density)
        return omega, coupling

    space = np.linspace(start, stop, num + 1, endpoint=True)
    omega_0, omega_1 = space[:-1], space[1:]
    ans = list(map(direct_quad, omega_0, omega_1))
    return ans


def generate_BCF(ph_parameters, bath_corr: Optional[Correlation] = None, beta=None):
    """
    Args:
        ph_parameters: [(frequency, coupling)]
    """

    coeff = []
    conj_coeff = []
    derivative = []
    for omega, g in ph_parameters:
        temp_factor = 1.0 / np.tanh(beta * omega / 2) if beta is not None else 1.0
        coeff.extend([g**2 / 2.0 * (temp_factor - 1.0), g**2 / 2.0 * (temp_factor + 1.0)])
        conj_coeff.extend([g**2 / 2.0 * (temp_factor + 1.0), g**2 / 2.0 * (temp_factor - 1.0)])
        derivative.extend([1.0j * omega, -1.0j * omega])

    if bath_corr is not None:
        coeff += list(bath_corr.coeff)
        conj_coeff += list(bath_corr.conj_coeff)
        derivative += list(bath_corr.derivative)

    corr = Correlation(
        k_max=len(coeff),
        beta=beta,
        coeff=coeff,
        conj_coeff=conj_coeff,
        derivative=derivative,
    )

    return corr
