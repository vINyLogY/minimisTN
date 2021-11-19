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


def generate_BCF(ph_parameters, beta=None):
    """
    Args:
        ph_parameters: [(frequency, coupling)]
    """
    dim = 2 * len(ph_parameters)
    coeff = []
    conj_coeff = []
    derivative = []
    for omega, g in ph_parameters:
        temp_factor = 1.0 / np.tanh(beta * omega / 2) if beta is not None else 1.0
        coeff.extend([g**2 / 2.0 * (temp_factor - 1.0), g**2 / 2.0 * (temp_factor + 1.0)])
        conj_coeff.extend([g**2 / 2.0 * (temp_factor + 1.0), g**2 / 2.0 * (temp_factor - 1.0)])
        derivative.extend([1.0j * omega, -1.0j * omega])
    corr = Correlation(
        k_max=dim,
        beta=beta,
        coeff=coeff,
        conj_coeff=conj_coeff,
        derivative=derivative,
    )
    return corr


class SpectralDensityFactory(object):

    @staticmethod
    def drude_lorentz(lambda_, omega):

        def _drude(w):
            c = 0.5
            return c * (lambda_ * w * omega) / (w**2 + omega**2)

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


if __name__ == '__main__':
    from minitn.lib.tools import figure, plt
    from minitn.lib.units import Quantity
    j_w = SpectralDensityFactory.bimodal_spectral_density(
        Quantity(2250, 'cm-1').value_in_au,
        Quantity(500, 'cm-1').value_in_au,
        Quantity(1250, 'cm-1').value_in_au,
        Quantity(50, 'cm-1').value_in_au)
    ans = linear_discretization(j_w, Quantity(2500, 'cm-1').value_in_au, 32)
    x, y = zip(*ans)
    x = list(map(lambda w: Quantity(w).convert_to('cm-1').value, x))
    for i, j in zip(x, y):
        msg = '{:.1f} & {:.5f} \\\\'.format(i, j * 1e5)
        print(msg)
    with figure():
        plt.plot(x, y, 'o')
        plt.show()
