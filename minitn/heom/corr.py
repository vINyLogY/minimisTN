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
