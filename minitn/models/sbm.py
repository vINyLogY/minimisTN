#!/usr/bin/env python
# coding: utf-8
r"""Functions and objects about system--bath interaction model::

    H = H_S + H_B + H_SB

    where H_S is charecterized by e and v;
    H_B: [w]
    H_SB: S, [(g, w)]

"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from itertools import filterfalse, count

from minitn.lib.backend import np
from scipy import linalg
from scipy.integrate import quad

from minitn.models.particles import Phonon
from minitn.models.bath import generate_BCF
from minitn.heom.hierachy import Hierachy

DTYPE = np.complex128


class SBM(object):

    def __init__(self, sys_ham, sys_op, ph_parameters, n_dims) -> None:
        """
        Args:
        ph_parameters: [(frequency, coupling)]
        n_dims: [int]
        """
        self.h = sys_ham
        self.op = sys_op
        self.ph_parameters = ph_parameters
        self.n_dims = n_dims
        return

    def wfn_h_list(self, sys_index, ph_indices):
        h_list = []
        h_list.append([(sys_index, -1.0j * self.h)])

        for ph_index, (omega, g), d in zip(ph_indices, self.ph_parameters, self.n_dims):
            ph = Phonon(d, omega)
            # hamiltonian ph part
            h_list.append([(ph_index, -1.0j * ph.hamiltonian)])
            # e-ph part
            op = ph.annihilation_operator + ph.creation_operator
            h_list.append([(ph_index, g * op), (sys_index, -1.0j * self.op)])

        return h_list

    def heom_h_list(self, sys_i, sys_j, ph_indices: list, beta=None):
        corr = generate_BCF(self.ph_parameters, beta=beta)
        n_tiers = list(np.repeat(self.n_dims, 2))
        diff = Hierachy(n_tiers, self.h, self.op, corr).diff()
        index_conv = list(ph_indices) + [sys_i, sys_j]
        h_list = []
        for term in diff:
            h_list.append([(index_conv[fst], snd) for fst, snd in term])

        return h_list
