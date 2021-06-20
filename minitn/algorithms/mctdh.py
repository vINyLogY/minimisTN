#!/usr/bin/env python
# coding: utf-8
r"""A Simple MCTDH Program, Based on PO-DVR

References
----------
.. [1] J. Chem. Phys. 119, 1289 (2003)
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import fftpack, linalg
from scipy.integrate import RK45
from scipy.sparse.linalg import LinearOperator, eigsh

from minitn.bases.dvr import PO_DVR
from minitn.lib import numerical
from minitn.lib.numerical import DavidsonAlgorithm, expection
from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import figure


class MCTDH(PO_DVR):
    r""" Structure of the wavefunction/state::

        n_0|   |   |n_p-1
          C_0 ... C_p-1
             \ | /
          m_0 \|/ m_p-1
               A

    and corresponding Hamiltonian:
    * Option 1::

          n_0/   \n_p-1
           h_0   h_p-1
        n_0/ \   / \n_p-1
               + r
               |
           -- ... --

    Note that the order of contraction is essential.
    """
    def __init__(self, conf_list, shape_list, fast=False, hbar=1., m_e=1., ):
        r"""
        N-dimensional DVR using sine-DVR for 1-D::

        Parameters
        ----------
        conf_list : [(float, float, int)]
        shape_list : [(int, int), ..., (int, int)]
            ``shape_list == [(n_1, m_1), ..., (n_p, m_p)]``, which correspoed
            to the structure of state tree.
        hbar : float, optional
        hbar : float, optional
        fast : bool, optional
        """
        super(MCTDH, self).__init__(
            conf_list, fast=fast, hbar=hbar, m_e=m_e
        )
        # shape[1] is m and shape[0] is n
        shape_a = [shape[1] for shape in shape_list]
        # add shape of A tensor to the end
        self.shape_list = shape_list + [shape_a]
        self.size_list = [np.prod(shape) for shape in self.shape_list]
        self.size = sum(self.size_list)

        self.h_terms = None
        self.mod_terms = None
        self.vec = None

    def gen_h_terms(self, extra=None, kinetic_only=False):
        r"""Use a simple seperated Hamiltonian operator::

            h_0 ... h_p-1
               \ | /
                \|/
                 + r

        Notice that r is a index rather than a tensor

        Parameters
        ----------
        extra : [[(int, float -> float)]]
        kinetic_only : bool

        Returns
        -------
        h_terms : [[(int, (n_i, n_i) ndarray)]]
            A list of Hamiltonian matrix. (with ``length == rank``)
        """
        dvr_list = self.dvr_list
        if kinetic_only:
            h_terms = [
                [(i, dvr.t_mat())] for i, dvr in enumerate(self.dvr_list)
            ]
        else:
            h_terms = [
                [(i, dvr.h_mat())] for i, dvr in enumerate(self.dvr_list)
            ]
        if extra is not None:
            for term in extra:
                t_i = [
                    (i, np.diag(func(np.asarray(self.grid_points_list[i]))))
                    for i, func in term
                ]
                h_terms.append(t_i)
        self.h_terms = h_terms
        return h_terms

    def update_mod_terms(self, vec=None, write=True):
        def _eval(op, vec):
            vec_h = np.conj(np.transpose(vec))
            mod_op = np.dot(vec_h, np.dot(op, vec))
            return mod_op

        mod_terms = []
        for term in self.h_terms:
            mod_terms.append([])
            for i, h in term:
                c = self.get_sub_vec(i, vec=vec)
                real, imag = c.real, c.imag
                eval_ = partial(_eval, h)
                h_real, h_imag = map(eval_, (real, imag))
                mod_h = h_real + h_imag
                t = (i, mod_h)
                mod_terms[-1].append(t)
        if write:
            self.mod_terms = mod_terms
        return mod_terms

    def init_state(self):
        r"""Form the initial vector according to shape list::

            n_0|   |   |n_p-1
              C_0 ... C_p-1
                 \ | /
              m_0 \|/ m_p-1
                   A

        Returns
        -------
        init : (self.size,) ndarray
            Formally, init = np.concatenate([C_0, ..., C_p-1, A], axis=None),
            where C_i is a (n_i * m_i,) ndarray, i <- {0, ..., p-1},
            A is a (M,) ndarray, and M = m_0 * ... * m_p-1, m_i < n_i.
        """
        dvr_list = self.dvr_list
        c_list = []
        for i, (_, m_i) in enumerate(self.shape_list[:-1]):
            _, v_i = dvr_list[i].solve(n_state=m_i)
            v_i = np.transpose(v_i)
            c_list.append(np.reshape(v_i, -1))
        vec_a = np.zeros(self.size_list[-1])
        vec_a[0] = 1.0
        vec_list = c_list + [vec_a]
        init = np.concatenate(vec_list, axis=None)
        self.vec = init
        self.update_mod_terms()
        return init

    def h_mat(self):
        """Formal Hamiltonian H acting on a MCTDH vector s. t.::

                     .
            i hbar |vec> = H |vec>

        Returns
        -------
            (self.size, self.size) LinearOperator
        """
        class _EffHamiltonian(LinearOperator):
            """
            Parameters
            ----------
            instance : MCTDH
                An MCTDH instance.
            """
            def __init__(self, instance):
                self.size = instance.size
                self.n_terms = len(instance.h_terms)
                self.term_func = instance.term_hamiltonian
                shape = [self.size] * 2
                super(_EffHamiltonian, self).__init__('d', shape)

            def _matvec(self, vec):
                ans = np.zeros_like(vec, dtype=complex)
                for i in range(self.n_terms):
                    ans += self.term_func(i, vec)
                return ans

        return _EffHamiltonian(self)

    def term_hamiltonian(self, r, vec):
        """
        Parameters
        ----------
        r : int
        vec : (self.size,) ndarray
        """
        logging.debug(__(
            'Hamiltonian term {}...', r
        ))
        h_term = self.h_terms[r]
        mod_term = self.mod_terms[r]
        steps = len(self.shape_list)
        ans = []
        for i in range(steps):
            v_i = self.get_sub_vec(i, vec)
            if i < steps - 1:
                h_list = [h for j, h in h_term if j == i]
                v_i = self._sp_op(i, v_i, h_list, mod_term)
            else:
                v_i = self._coeff_op(v_i, mod_term)
            v_i = np.reshape(v_i, -1)
            ans.append(v_i)
        ans = np.concatenate(ans, axis=None)
        return ans

    def _sp_op(self, i, mat, h_list, mod_term, err=1.e-6):
        if not h_list:
            return np.zeros((mat.shape))

        logging.debug(__(
            '> OP on mat {}...', i
        ))

        n, m = mat.shape
        partial_transform = self._partial_transform
        a = self.get_sub_vec(-1)
        a_h = np.conj(a)
        density = self._partial_product(i, a, a_h)
        inv_density = linalg.inv(density + np.identity(m) * err)
        sp = self.get_sub_vec(i)
        sp_h = np.conj(np.transpose(sp))
        projection = np.identity(n) - np.dot(sp, sp_h)

        tmp = partial_transform(i, a, mat)
        for mat_j in h_list:
            tmp = partial_transform(i, tmp, mat_j)
        for j, mat_j in mod_term:
            if j != i:
                tmp = partial_transform(j, tmp, mat_j)
        tmp = self._partial_product(i, tmp, a_h)
        ans = np.dot(projection, np.dot(tmp, inv_density))

        return ans

    def _coeff_op(self, tensor, mod_term):
        logging.debug(__(
            '> OP on A tensor...'
        ))
        for i, mat in mod_term:
            tensor = MCTDH._partial_transform(i, tensor, mat)
        return tensor

    @staticmethod
    def _partial_product(i, a, b):
        r"""
        Parameters
        ----------
        i : {int, None}
        a : (..., n, ...) ndarray
        b : (..., m, ...) ndarray

        Returns
        -------
        mat : (n, m) ndarray
            Or return a float if i is None.
        """
        if i is None:
            a = np.reshape(a, -1)
            b = np.reshape(b, -1)
        else:
            n = a.shape[i]
            m = b.shape[i]
            a = np.moveaxis(a, i, 0)
            a = np.reshape(a, (n, -1))
            b = np.moveaxis(b, i, -1)
            b = np.reshape(b, (-1, m))
        mat = np.dot(a, b)
        return mat

    @staticmethod
    def _partial_transform(i, tensor, mat):
        r"""
        Parameters
        ----------
        i : int
        tensor : (..., m_i, ...) ndarray
        mat : (n_i, m_i) ndarray

        Returns
        -------
        tensor : (..., n_i, ...) ndarray
        """
        tensor_shape = list(tensor.shape)
        shape = tensor_shape[:i] + tensor_shape[i + 1:] + [mat.shape[0]]
        v_i = np.swapaxes(tensor, -1, i)
        v_i = np.reshape(v_i, (-1, mat.shape[1]))
        v_i = np.array(list(map(mat.dot, v_i)))
        v_i = np.reshape(v_i, shape)
        v_i = np.swapaxes(v_i, -1, i)
        return v_i

    def get_sub_vec(self, i, vec=None):
        """Get C_i mat or A tensor from MCTDH vec.

        Parameters
        ----------
        i : int
            If 0 <= i < len(self.shape_list) - 1, return C_i mat, else return
            A tensor.
        """
        if vec is None:
            vec = self.vec
        size_list = self.size_list
        i = i % len(size_list)
        start = sum(size_list[:i])
        end = sum(size_list[:i + 1])
        sub = vec[start:end]
        return np.reshape(sub, self.shape_list[i])

    def energy_expection(self, vec=None):
        if vec is None:
            vec = self.vec
        a_tensor = self.get_sub_vec(-1, vec)
        mod_terms = self.update_mod_terms(vec=vec, write=False)
        ans = 0.
        for r, term in enumerate(mod_terms):
            h_a = self._coeff_op(a_tensor, term)
            h_a = np.reshape(h_a, -1)
            a_h = np.conj(np.reshape(a_tensor, -1))
            ans += np.dot(a_h, h_a)
        return ans

    def propagation(self, renormalize=False, *args, **kwargs):
        func = super(MCTDH, self).propagation
        __doc__ = func.__doc__
        dot = np.dot

        def _updater(instance):
            self.vec = instance.y
            self.update_mod_terms()
            return

        def _normalizer(vec):
            ans = []
            for i in range(self.rank + 1):
                vec_i = self.get_sub_vec(i, vec)
                vec_i = np.reshape(vec_i, -1)
                if i == self.rank:
                    std_norm = sqrt(1.0)
                else:
                    std_norm = sqrt(self.shape_list[i][1])
                norm = (
                    linalg.norm(vec_i) / std_norm
                )
                logging.debug(__(
                    'norm {}: {}', i, norm
                ))
                ans.append(vec_i / norm)
            ans = np.concatenate(ans, axis=None)
            return ans

        normalizer = _normalizer if renormalize else None

        return func(
            *args, updater=_updater, normalizer=normalizer, **kwargs
        )

    def autocorrelation(self, *args, **kwargs):
        func = super(MCTDH, self).autocorrelation
        __doc__ = func.__doc__
        dot = np.dot
        get_sub_vec = self.get_sub_vec
        partial_transform = self._partial_transform

        def _dot(v1, v2):
            t1 = np.reshape(get_sub_vec(-1, v1), -1)
            t2 = np.reshape(get_sub_vec(-1, v2), -1)
            ans = dot(t1, t2)
            return ans

        return func(*args, dot=_dot, **kwargs)


# EOF
