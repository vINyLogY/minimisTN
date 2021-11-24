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

    def __init__(self, sys_ham, sys_op, ph_parameters, ph_dims, bath_corr=None, bath_dims=None) -> None:
        """
        Args:
        ph_parameters: [(frequency, coupling)]
        bcf_parameters: [(coeff, coeff_conj, derivative)]
        n_dims: [int]
        """
        self.h = sys_ham
        self.op = sys_op
        self.ph_parameters = ph_parameters
        self.ph_dims = ph_dims
        self.bath_corr = bath_corr
        self.bath_dims = bath_dims
        return

    def wfn_h_list(self, sys_index, ph_indices):
        h_list = []
        h_list.append([(sys_index, -1.0j * self.h)])

        for ph_index, (omega, g), d in zip(ph_indices, self.ph_parameters, self.ph_dims):
            ph = Phonon(d, omega)
            # hamiltonian ph part
            h_list.append([(ph_index, -1.0j * ph.hamiltonian)])
            # e-ph part
            op = ph.annihilation_operator + ph.creation_operator
            h_list.append([(ph_index, g * op), (sys_index, -1.0j * self.op)])

        return h_list

    def heom_h_list(self, sys_i, sys_j, bath_indices: list, beta=None):
        corr = generate_BCF(self.ph_parameters, bath_corr=self.bath_corr, beta=beta)
        n_tiers = list(np.repeat(self.ph_dims, 2))
        if self.bath_dims is not None:
            n_tiers += self.bath_dims
        diff = Hierachy(n_tiers, self.h, self.op, corr).h_list(sys_i, sys_j, bath_indices)
        return diff
