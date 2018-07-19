#!/usr/bin/env python2
# coding: utf-8
"""(1-D) CAS library by sympy
"""
from __future__ import division

import logging
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import sympy as sym

sym.init_printing()
x, y, z = sym.symbols('x y z')


class BasisFunction(object):
    def particle_in_box(self, j, L, x0):
        def _phi(x):
            phi = np.where(
                np.logical_and(x0 < x, x < x0 + L),
                np.sqrt(2. / L) * np.sin(j * np.pi * (x - x0) / L),
                0)
            return phi
        return _phi

    def harmonic_oscillator(n, k=1., m=1., hbar=1.):
        psi = harmonic_oscillator(n, k=k, m=m, hbar=hbar)
        psi = lambdify(psi)
        return psi


class PotentialFunction(object):
    def square_well(self, depth=1., width=1., x0=0., v0=0.):
        r"""Returns a function of a single variable V(x).

            (x0, v0+depth)    (x0+width, v0+depth)
                     ----+    +----
                         |    |
                (x0, v0) +----+ (x0+width, v0)
        """
        def _v(x):
            if x0 < x and x < x0 + width:
                return v0
            else:
                return v0 + depth

        return _v

    def w_well(self, d0=5., a=1.):
        r"""
                \ (0,d1) /
                 \  /\  /
            (-a,0)\/  \/(a, 0)
        """
        return lambda x: (d0 / a ** 4) * (x ** 2 - a ** 2) ** 2

    def sho(self, k=1., x0=0.):
        """Return a one-dimensional harmonic oscillator potential V(x)
        with wavenumber k.
        """
        return lambda x: 0.5 * (k * (x - x0)) ** 2


def id_op():
    def _id(y):
        return y
    return _id


def prod_op(m):
    def _prod(y):
        return m * y
    return _prod


def diff(n):
    def _diff(y):
        x = sym.symbols('x')
        return sym.diff(y, x, n)
    return _diff


def matrix_element(i, op, j, cut_off=None, num_prec=None):
    x = sym.symbols('x')
    prod = i(x) * op(j(x))
    interval = cut_off
    if num_prec is None:
        if interval is None:
            interval = (-sym.oo, sym.oo)
        mel = sym.integrate(prod, (x, interval[0], interval[1]))
        mel = mel.evalf()
    else:
        if interval is None:
            interval = (-1.e4, 1.e4)
        prod = sym.lambdify(x, prod, modules='numpy')
        mel = quadrature(prod, interval[0], interval[1], num_prec)

    return mel


def quadrature(func, start, stop, num_prec):
    x = np.linspace(start, stop, num=num_prec + 1)
    fx = func(x)
    delta = (stop - start) / num_prec
    area = (fx[:-1] + fx[1:]) * delta / 2.
    quad = np.sum(area)
    return quad


def lambdify(f):
    x = sym.symbols('x')
    _f = sym.lambdify(x, f(x), modules='numpy')
    return _f


def matrix_repr(op, basis, cut_off=None, num_prec=None):
    # op is a function now.
    n_dims = len(basis)
    A = np.zeros((n_dims, n_dims))
    x = sym.symbols('x')
    for i in range(n_dims):
        for j in range(i + 1):
            mel = matrix_element(
                basis[i], op, basis[j],
                cut_off=cut_off, num_prec=num_prec)
            A[i, j] = mel
            A[j, i] = mel
    return A


def particle_in_box(k, L, x0):
    # not a piecewise function, set cut_off manually
    def _phi(y):
        phi = sym.sqrt(2. / L) * sym.sin(k * sym.pi * (y - x0) / L)
        return phi
    return _phi


def hermite_polynomials(n):
    def _h(y):
        x = sym.symbols('x')
        h = (-1) ** n * sym.exp(x ** 2) \
            * sym.diff(sym.exp(- x ** 2), x, n)
        h = sym.simplify(h)
        return h.subs(x, y)
    return _h


def harmonic_oscillator(n, k=1., m=1., hbar=1.):
    def _psi(y):
        factor = m * k / hbar
        psi = sym.expand(
            sym.root(factor, 4) * (hermite_polynomials(n))(factor * y) /
            sym.sqrt(2. ** n * math.factorial(n)))
        psi = psi * sym.exp(- factor / 2. * y ** 2) / sym.root(sym.pi, 4)

        return psi
    return _psi


def main():
    for n in range(6):
        print (harmonic_oscillator(n))(x)
    return


if __name__ == '__main__':
    main()
