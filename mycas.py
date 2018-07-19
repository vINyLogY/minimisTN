#!/usr/bin/env python2
# coding: utf-8
"""(1-D) CAS library by sympy
"""
from __future__ import division

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import sympy as sym

sym.init_printing()
x, y, z = sym.symbols('x y z')


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
