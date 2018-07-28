#!/usr/bin/env python2
# coding: utf-8
"""Playground
"""
from __future__ import division

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sympy as sym
from scipy.sparse.linalg import LinearOperator

import _context
import minitn


def tenserize(shape):
    shape = list(shape)
    if shape == []:
        return [[]]
    else:
        ans = []
        for x in range(shape[0]):
            sub = tenserize(shape[1:])
            ans += [[x] + xs for xs in sub]
        return ans


def subindex(N, shape):
    rank = len(shape)
    prods = [1]
    for i in range(rank - 1, 0, -1):
        prods.append(prods[-1] * shape[i])
    sub = []
    for i in range(rank):
        base = prods.pop()
        sub.append(N // base)
        N = N % base
    return sub


a = (2, 3, 4)

logging.debug('mess!')
logging.warning('Warning!')

for i in range(np.prod(a)):
    b = tenserize(a)[i]
    c = subindex(i, a)
    print('b: {}; c: {}'.format(b, c))
