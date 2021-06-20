#!/usr/bin/env python
# coding: utf-8
"""
Conversion:
    rho[i, j, n_0, ..., n_(k-1)]
"""

from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np

from minitn.lib.tools import __

class Hierachy(object):
    def __init__(self, init_rho, k_dims, sys_hamiltonian, sys_op):
        assert init_rho.ndim == 2 and init_rho.shape[0] == init_rho.shape[1]
        self.k_dims = k_dims
        self.k_max = len(k_dims)
        self.init_rho = init_rho
        self.op = sys_op
        self.hamiltonian = sys_hamiltonian

    def eom(self):
        pass



