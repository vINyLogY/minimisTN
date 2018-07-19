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

x = sym.symbols('x')


def id_op():
    def _id(y):
        x = sym.symbols('x')
        return x.subs(x, y)
    return _id


def prod_op(m):
    def _prod(y):
        x = sym.symbols('x')
        return (m * x).subs(x, y)
    return _prod


def diff(n):
    def _diff(y):
        x = sym.symbols('x')
        return sym.diff(y, x, n)
    return _diff


def matrix_element(i, op, j):
    x = sym.symbols('x')
    prod = i(x) * op(j(x))
    mel = sym.integrate(prod, (x, -sym.oo, sym.oo)).evalf()
    return mel


def lambdify(f):
    x = sym.symbols('x')
    _f = sym.lambdify(x, f(x), modules='numpy')
    return _f


def matrix_repr(op, basis):
    # op is a function now.
    n_dims = len(basis)
    A = np.zeros((n_dims, n_dims))
    x = sym.symbols('x')
    for i in range(n_dims):
            for j in range(i + 1):
                mel = matrix_element(basis[i], op, basis[j])
                Q[i, j] = mel
                Q[j, i] = mel
    return A


def particle_in_box(j, L, x0):
    def _phi(y):
        x = sym.symbols('x')
        phi = sym.Piecewise(
            (sym.sqrt(2. / L) * sym.sin(j * sym.pi * (x - x0) / L),
                (x0 < x) & (x < x0 + L)),
            (0, True))
        return phi.subs(x, y)
    return _phi
