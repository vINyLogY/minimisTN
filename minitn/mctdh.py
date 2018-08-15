#!/usr/bin/env python
# coding: utf-8
r"""A Simple DVR Program (n-D)

References
----------
.. [1] J. Chem. Phys. 119, 1289 (2003)
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import fftpack
from scipy.integrate import RK45
from scipy.sparse.linalg import LinearOperator, eigsh

from minitn.dvr import PO_DVR
from minitn.lib import numerical
from minitn.lib.numerical import DavidsonAlgorithm, expection
from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import figure


class MCTDH(PO_DVR):
    def __init__(self, conf_list, shape_list, fast=False, hbar=1., m_e=1., ):
        """
        N-dimensional DVR using sine-DVR for 1-D.

        Parameters
        ----------
        conf_list : [(float, float, int)]
        shape_list : [(float, float), ..., (float, float)]
            ``shape_list == [(n_1, m_1), ..., (n_p, m_p)]``, which correspoed
            to the structure of state tree.
        hbar : float, optional
        hbar : float, optional
        fast : bool, optional
        """
        super(MCTDH, self).__init__(
            conf_list, fast=fast, hbar=hbar, m_e=m_e
        )
        self.shape_list = shape_list
        # shape[1] is m and shape[0] is n
        self.shape_m = [shape[1] for shape in shape_list]
        self.shape_n = [shape[0] for shape in shape_list]
        size_c_list = [np.prod(shape) for shape in shape_list]
        self.size_list = [np.prod(shape_m)] + size_c_list
        self.size = sum(self.size_list)

        self.h_terms = None
        self.vec = None

    def gen_h_terms(self, extra=None):
        """
        Parameters
        ----------
        extra : [[float -> float]]

        Returns
        -------
        mpo : [(r, n_i, n_i) ndarray]
            A list of MPO matrix. (with ``length == rank``)
        """
        h_list = [dvr.h_mat() for dvr in self.dvr_list]
        eye_list = [eye(h.shape[0]) for h in h_list]
        h_terms = []
        for i in range(self.rank):
            h_terms.append([])
            for j in range(self.rank):
                op = h_list[j] if j == i else eye_list[j]
                h_terms[i].append(op)
        if extra is not None:
            for term in extra:
                zipped = zip(term, self.dvr_list)
                t_i = [np.diag(func(dvr.grid_points)) for func, dvr in zipped]
                mpo.extend(t_i)
        self.h_term = h_terms
        return self.h_terms

    def _init_state(self):
        """Form the initial vector according to shape list.
        Returns
        -------
        init : (self.size,) ndarray
            Formally, init = np.append(A, c_1, ..., c_p), where
            A is a (m_1 * ... * m_p,) ndarray,
            c_i is a (n_i * m_i,) ndarray, i <- {1, ... p},
            and N = m_1 * ... * m_p, m_i < n_i.
        """
        dvr_list = self.dvr_list
        c_list = []
        for i, shape in enumerate(self.shape_list):
            m_i = shape[1]
            _, v_i = dvr_list[i].solve(n_state=m_i)
            v_i = np.transpose(v_i)
            c_list.append(np.reshape(v_i, -1))
        vec_a = np.zeros(self.size_list[0])
        vec_a[0] = 1.0
        vec_list = [vec_a] + c_list
        init = np.append(vec_list)
        self.vec = init
        return init

    def h_mat(self):
        class _EffHamiltonian(LinearOperator):
            def __init__(self, size_list, op_list):
                self.size_list = size_list
                self.op_list = op_list
                shape = [sum(size_list)] * 2
                super(_EffHamiltonian, self).__init__('d', shape)

            def _matvec(self, vec):
                ans_list = []
                zipped = zip(self.size_list, self.op_list)
                for s, op in zipped:
                    vec_i = vec[:s]
                    vec = vec[s:]
                    ans_list.append(op.dot(vec_i))
                ans = np.append(*ans_list)
                return ans

        op_list = [self._coeff_op()]
        for i in range(self.rank):
            op_list.append(self._sp_op(i))
        return self._EffHamiltonian(self.size_list, op_list)

    def _coeff_op(self):
        class _Coeff(LinearOperator):
            def __init__(self, h_terms, c_list):
                self.shape_m = [c_i.shape[1] for c_i in c_list]
                shape = [np.prod(shape_m)] * 2
                super(_Coeff, self).__init__('d', shape)

                self.h_terms = h_terms
                self.c_list = c_list
                new_terms = []
                for i, term in enumerate(self.h_terms):
                    new_terms.append([])
                    zipped = zip(term, c_list)
                    for h, c in zipped:
                        c_h = np.conj(np.transpose(c))
                        t = np.dot(c_h, np.dot(h, c))
                        new_terms[i].append(t)
                self.new_terms = new_terms

            def _matvec(self, vec):
                shape_in = self.shape_m
                ans = np.zeros(shape)
                for term in self.new_terms:
                    v = vec
                    for i, h_i in enumerate(term):
                        shape_out = shape[:i] + shape[i + 1:] + [shape[i]]
                        v = np.swapaxes(v, -1, i)
                        v = np.reshape(v, (-1, shape_in[i]))
                        v = np.array(list(map(h_i.dot, v)))
                        v = np.reshape(v, shape_out)
                        v = np.swapaxes(v, -1, i)
                    ans += v
                ans = np.reshape(ans, -1)
                return ans

        h_terms = self.h_terms
        c_list = self._get_c_list()
        return _Coeff(h_terms, c_list)

    def _get_c_list(self):
        vec = self.vec[self.size_list[0]:]
        sizes = self.size_list[1:]
        zipped = zip(sizes, self.shape_list)
        c_list = []
        for s, shape in zipped:
            vec_i = vec[:s]
            vec = vec[s:]
            c_i = np.reshape(vec_i, shape)
            c_list.append(c_i)
        return c_list

    def _sp_op(self, k):
        class _Sp(LinearOperator):
            def __init__(self, k, h_terms, coeff, c_list):
                self.shape_k = c_list[k].shape
                self.shape_m = [c_i.shape[1] for c_i in c_list]
                self.shape_nk = c_list[k].shape[0]
                self.k = k
                shape = [np.prod(shape_m)] * 2
                super(_Sp, self).__init__('d', shape)

                self.h_terms = h_terms
                self.coeff = coeff
                self.c_list = c_list

            def _matvec(self, vec):
                # construct with mean field <H>^k and projection
                k = self.k
                new_terms = []
                for i, term in enumerate(self.h_terms):
                    new_terms.append([])
                    zipped = zip(term, c_list)
                    for h, c in zipped:
                        if i != k:
                            c_h = np.conj(np.transpose(c))
                        else:
                            c_h = self._projection()
                        t = np.dot(c_h, np.dot(h, c))
                        new_terms[j].append(t)

                # construct with coeff
                shape_in = self.shape_m
                ans = np.zeros(shape)
                for term in new_terms:
                    v = self.coeff
                    for i, h_i in enumerate(term):
                        shape_out = shape[:i] + shape[i + 1:] + [-1]
                        v = np.swapaxes(v, -1, i)
                        v = np.reshape(v, (-1, shape_in[i]))
                        v = np.array(list(map(h_i.dot, v)))
                        v = np.reshape(v, shape_out)
                        v = np.swapaxes(v, -1, i)
                    ans += v

                # construct with coeff_h and inv_density
                ans = np.swapaxes(ans, -1, _i)
                coeff_h = np.conj(np.swapaxes(coeff, 0, _i))
                ans = np.dot(coeff_h, ans)
                ans = np.dot(self._inv_density(), ans)
                return ans

            def _projection(self):
                # TODO
                pass

            def _inv_density(self):
                # TODO
                pass

        h_op = super(MCTDH, self).h_mat()
        coeff = np.reshape(self.vec[:self.size_list[0]], self.shape_m)
        c_list = self._get_c_list()
        return _Sp(i, h_op, c_list)


# Helper functions
def _sp_transform(c_list, coeff):
    r"""Do the contraction::

        n_1|   |   |n_p
          c_1 ... c_p
            \  |  /
         m_1 \ | / m_p
             coeff

    Parameters
    ----------
    c_list : [(n_i, m_i) ndarray]
        SPF matrix.
    coeff : (m_1, ..., m_p) ndarray
        Coefficient tensor in SPF basis.

    Returns
    -------
    coeff : (n_1, ..., n_p) ndarray
        Coefficient tensor in primitive basis.
    """
    # shape will be update to the right shape of vec
    # during iteration
    shape = coeff.shape
    for i, c_i in enumerate(c_list):
        shape = shape[:i] + [c_i.shape[0]] + shape[i + 1:]
        shape_i = shape[:i] + shape[i + 1:] + [c_i.shape[0]]
        v_i = np.swapaxes(coeff, -1, i)
        v_i = np.reshape(v_i, (-1, c_i.shape[1]))
        v_i = np.array(list(map(c_i.dot, v_i)))
        v_i = np.reshape(tmp, shape_i)
        coeff = np.swapaxes(v_i, -1, i)
    return coeff
