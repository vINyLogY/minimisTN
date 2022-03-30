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
from minitn.heom.corr import Correlation
from minitn.heom.hierachy import Hierachy

DTYPE = np.complex128


class SpinBoson(object):

    def __init__(self,
                 sys_ham,
                 sys_op,
                 ph_parameters,
                 ph_dims,
                 bath_corr=None,
                 bath_dims=None) -> None:
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

        self.bath_corr = bath_corr if bath_corr is not None else []
        self.bath_dims = bath_dims if bath_dims is not None else []
        return

    def wfn_h_list(self, sys_index, ph_indices):
        h_list = []
        h_list.append([(sys_index, -1.0j * self.h)])

        for ph_index, (omega, g), d in zip(ph_indices, self.ph_parameters,
                                           self.ph_dims):
            ph = Phonon(d, omega)
            # hamiltonian ph part
            h_list.append([(ph_index, -1.0j * ph.hamiltonian)])
            # e-ph part
            op = ph.annihilation_operator + ph.creation_operator
            h_list.append([(ph_index, g * op), (sys_index, -1.0j * self.op)])

        return h_list

    def do_l_list(self, sys_i, sys_j, ph_is, ph_js):
        """SPDO Liouvillian
        """
        # FIXME
        l_list = []
        l_list.append([(sys_i, -1.0j * self.h)])
        l_list.append([(sys_j, 1.0j * self.h)])

        for ph_i, (omega, g), d in zip(ph_is, self.ph_parameters,
                                       self.ph_dims):
            ph = Phonon(d, omega)
            # hamiltonian ph part
            l_list.append([(ph_i, -1.0j * ph.hamiltonian)])
            # e-ph part
            op = ph.annihilation_operator + ph.creation_operator
            l_list.append([(ph_i, g * op), (sys_i, -1.0j * self.op)])

        for ph_j, (omega, g), d in zip(ph_js, self.ph_parameters,
                                       self.ph_dims):
            ph = Phonon(d, omega)
            # hamiltonian ph part
            l_list.append([(ph_j, 1.0j * ph.hamiltonian)])
            # e-ph part
            op = ph.annihilation_operator + ph.creation_operator
            l_list.append([(ph_j, g * op), (sys_j, 1.0j * self.op)])

        return l_list

    def heom_h_list(self,
                    sys_i,
                    sys_j,
                    bath_indices: list = None,
                    beta=None,
                    basis=None,
                    scale=1.0):
        if bath_indices is None:
            bath_indices = []
        corr = generate_BCF(self.ph_parameters,
                            bath_corr=self.bath_corr,
                            beta=beta)
        n_tiers = list(np.repeat(self.ph_dims, 2))
        n_tiers += self.bath_dims

        heom = Hierachy(n_tiers, self.h, self.op, corr, basis=basis)
        heom.scale = scale
        diff = heom.h_list(sys_i, sys_j, bath_indices)

        return diff

    def heom_h_list2(self, sys_i, sys_j, bath_indices: list = None, beta=None):
        if bath_indices is None:
            bath_indices = []
        corr = generate_BCF(self.ph_parameters,
                            bath_corr=self.bath_corr,
                            beta=beta)
        n_tiers = list(np.repeat(self.ph_dims, 2))
        n_tiers += self.bath_dims
        heom = Hierachy(n_tiers, self.h, self.op, corr)
        heom.diff_type = 2
        diff = heom.h_list(sys_i, sys_j, bath_indices)
        return diff

    def dense_h(self,
                sys_i,
                sys_j,
                ph_indices: list,
                bath_indices=None,
                beta=None,
                f_type=None):
        """dimension convension: i, j, vec[bath]
        """
        if bath_indices is None:
            bath_indices = []
        indices = [sys_i, sys_j] + list(ph_indices) + list(bath_indices)
        shape = list(np.shape(self.h)) + list(np.repeat(
            self.ph_dims, 2)) + list(self.bath_dims)
        diff = self.heom_h_list(sys_i, sys_j, ph_indices, bath_indices, beta,
                                f_type)

        ii = np.reshape(np.identity(np.prod(shape), dtype=complex), shape * 2)
        h_tensor = np.zeros_like(ii, dtype=complex)
        # order = len(shape)

        for term in diff:
            for k, array in term:
                cumm = np.tensordot(ii, array, ([indices.index(k)], [0]))
                cumm = np.moveaxis(cumm, -1, indices.index(k))
                h_tensor += cumm

        return np.reshape(h_tensor, (np.prod(shape), np.prod(shape)))
