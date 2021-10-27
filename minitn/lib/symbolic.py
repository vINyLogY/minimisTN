#!/usr/bin/env python2
# coding: utf-8
"""Symbolic objects and methods.
"""
from __future__ import absolute_import, division

import logging
import math


from minitn.lib.backend import np
import sympy as sym

if __debug__:
    sym.init_printing()

x, y, z = sym.symbols('x y z')


class BasisFunction(object):
    @staticmethod
    def particle_in_box(k, L, x0):
        # not a piecewise function, set cut_off manually
        def _phi(y):
            phi = sym.sqrt(2. / L) * sym.sin(k * sym.pi * (y - x0) / L)
            return phi
        return _phi

    @staticmethod
    def harmonic_oscillator(n, k=1., m=1., hbar=1.):
        def _psi(y):
            factor = m * k / hbar
            hermite = Polynomial.hermite(n)
            psi = sym.expand(
                sym.root(factor, 4) * hermite(factor * y) /
                sym.sqrt(2. ** n * math.factorial(n))
            )
            psi = psi * sym.exp(- factor / 2. * y ** 2) / sym.root(sym.pi, 4)
            return psi
        return _psi


class Polynomial(object):
    @staticmethod
    def hermite(n):
        def _h(y):
            x = sym.symbols('x')
            h = (-1) ** n * sym.exp(x ** 2) * sym.diff(sym.exp(- x ** 2), x, n)
            h = sym.simplify(h)
            return h.subs(x, y)
        return _h


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
    from minitn.lib.numerical import quadrature
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


def matrix_repr(op, basis, cut_off=None, num_prec=None):
    # op is a function.
    n_dims = len(basis)
    A = np.zeros((n_dims, n_dims))
    x = sym.symbols('x')
    for i in range(n_dims):
        for j in range(i + 1):
            mel = matrix_element(
                basis[i], op, basis[j], cut_off=cut_off, num_prec=num_prec
            )
            A[i, j] = mel
            A[j, i] = mel
    return A


def lambdify(f):
    x = sym.symbols('x')
    _f = sym.lambdify(x, f(x), modules='numpy')
    return _f


def main():
    for n in range(6):
        print (BasisFunction.harmonic_oscillator(n))(x)
    return


if __name__ == '__main__':
    main()
