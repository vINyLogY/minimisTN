#!/usr/bin/env python2
# coding: utf-8
"""Numerical objects and methods.
"""
from __future__ import division

import logging
import math

import numpy as np
import scipy.linalg


class BasisFunction(object):
    @staticmethod
    def particle_in_box(j, L=1., x0=0.):
        def _phi(_x):
            phi = np.where(
                np.logical_and(x0 < _x, _x < x0 + L),
                np.sqrt(2. / L) * np.sin(j * np.pi * (_x - x0) / L),
                0
            )
            return phi
        return _phi

    @staticmethod
    def harmonic_oscillator(n, k=1., m=1., hbar=1.):
        from minitn.lib import symbolic
        psi = symbolic.BasisFunction.harmonic_oscillator(
            n, k=k, m=m, hbar=hbar
        )
        psi = symbolic.lambdify(psi)
        return psi


class PotentialFunction(object):
    @staticmethod
    def linear_corr(c=0.01):
        def _v(x):
            v = c * x[0] * x[1]
            return v
        return _v

    @staticmethod
    def square_well(depth=1., width=1., x0=0., v0=0.):
        r"""Returns a function of a single variable V(x).::

            (x0, v0+depth)    (x0+width, v0+depth)
                     ----+    +----
                         |    |
                (x0, v0) +----+ (x0+width, v0)

        """
        def _v(x):
            ans = np.where(
                np.logical_and(x0 < x, x < x0 + width),
                v0, v0 + depth)
            return ans
        return _v

    @staticmethod
    def w_well(d0=5., a=1.):
        r"""Double well potential energy.::

                \ (0,d1) /
                 \  /\  /
            (-a,0)\/  \/(a, 0)

        """
        return lambda x: (d0 / a ** 4) * (x ** 2 - a ** 2) ** 2

    @staticmethod
    def sho(k=1., x0=0.):
        """Return a one-dimensional harmonic oscillator potential V(x)
        with wavenumber k.
        """
        return lambda x: 0.5 * (k * (x - x0)) ** 2


def quadrature(func, start, stop, num_prec):
    x = np.linspace(start, stop, num=num_prec + 1)
    fx = func(x)
    delta = (stop - start) / num_prec
    area = (fx[:-1] + fx[1:]) * delta / 2.
    quad = np.sum(area)
    return quad


class DavidsonAlgorithm(object):
    """Davidson algorithm.

    Parameters
    ----------
    matvec : (N,) ndarray -> (N,) ndarray
    init_vecs : [(N,) ndarray]
        list containing k initial vectors
    n_vals : int
        Number of eigenvalues to be calculate (sorted from small to large,
        n_vals <= k).
    """
    tol = 1.e-8
    max_cycle = 50
    max_space = 12
    lin_dep_lim = 1.e-14

    @classmethod
    def config(cls, **kwargs):
        """Global configuration.

        Parameters
        ----------
        tol : float, optional
            Tolerance of energy.
        max_cycle : int, optional
        max_space : int, optional
        lin_dep_lim : float, optional
        """
        for key, value in kwargs.items():
            try:
                setattr(cls, key, value)
            except AttributeError:
                logging.warning('No configuration "{}"!'.format(key))
        return

    def __init__(self, matvec, init_vecs, n_vals=1, precondition=None):
        self._matvec = matvec
        self._n_vals = n_vals
        self._precondition = precondition
        self._trial_vecs = list(init_vecs)
        self._search_space = []
        self._column_space = []
        self._submatrix = np.zeros([self.max_space] * 2)

        self._last_ritz_vals = None
        self._ritz_vals = None
        self._ritz_vecs = None
        self._col_ritz_vecs = None
        self._residuals = None
        self._residual_norms = None
        self._diag = None

        self.eigvals = None
        self.eigvecs = None

    def kernel(self):
        """Run Davidson algorithm.

        Returns
        -------
        eigvals : (self.n_vals,) ndarray
        eigvecs : [(n,) ndarray]
        """
        for cycle in range(self.max_cycle):
            self._orthonormalize(use_svd=True)
            self._extend_space()
            self._calc_ritz_pairs()
            norms = self._calc_residual_norms()

            if self._last_ritz_vals is not None:
                n = min(len(self._last_ritz_vals), len(self._ritz_vals))
                diff_ritz_vals = np.abs(
                    self._last_ritz_vals[:n] - self._ritz_vals[:n]
                )
                if (max(norms) ** 2 < self.tol and
                        max(diff_ritz_vals) < self.tol):
                    break

            self._last_ritz_vals = self._ritz_vals
            self._calc_trial_vecs()
            next_space = len(self._trial_vecs) + len(self._search_space)
            if next_space > self.max_space:
                logging.debug('Search space too large, restart.')
                self.__init__(
                    matvec=self._matvec, init_vecs=self._trial_vecs,
                    n_vals=self._n_vals
                )

        if cycle + 1 >= self.max_cycle:
            logging.warning('Not converged!')
        self.eigvals = self._ritz_vals
        self.eigvecs = self._ritz_vecs
        return self.eigvals, self.eigvecs

    def _orthonormalize(self, use_svd=True):
        if use_svd and self._trial_vecs:
            trial_mat = np.transpose(np.array(self._trial_vecs))
            trial_mat = scipy.linalg.orth(trial_mat)
            self._trial_vecs = list(np.transpose(trial_mat))
        else:
            vecs = []
            for i, vec_i in enumerate(self._trial_vecs):
                for j, vec_j in enumerate(vecs):
                    vec_i -= vec_j * np.dot(vec_j.conj(), vec_i)
                norm = scipy.linalg.norm(vec_i)
                if norm > 1.e-7:
                    vecs.append(vec_i / norm)
            self._trial_vecs = vecs
        return self._trial_vecs

    def _extend_space(self):
        head = len(self._search_space)
        self._search_space += self._trial_vecs
        self._column_space += map(self._matvec, self._trial_vecs)
        tail = len(self._search_space)
        v, a_v = self._search_space, self._column_space
        for i in range(head):
            for j in range(head, tail):
                self._submatrix[i, j] = np.dot(np.conj(v[i]), a_v[j])
                self._submatrix[j, i] = np.conj(self._submatrix[i, j])
        for i in range(head, tail):
            for j in range(head, i):
                self._submatrix[i, j] = np.dot(np.conj(v[i]), a_v[j])
                self._submatrix[j, i] = np.conj(self._submatrix[i, j])
            self._submatrix[i, i] = np.dot(np.conj(v[i]), a_v[i])
        return self._submatrix[:tail, :tail]

    def _calc_ritz_pairs(self):
        def _trans_vec(vec, basis):
            new_vec = sum(map(lambda x, y: x * y, vec, basis))
            return new_vec

        n_space = len(self._search_space)
        n_state = min(n_space, self._n_vals)
        self._ritz_vals, v = scipy.linalg.eigh(
            self._submatrix[:n_space, :n_space], eigvals=(0, n_state - 1)
        )
        v = np.transpose(v)
        self._ritz_vecs = [_trans_vec(v_i, self._search_space) for v_i in v]
        self._col_ritz_vecs = [_trans_vec(v_i, self._column_space)
                               for v_i in v]
        return self._ritz_vals, self._ritz_vecs

    def _calc_residual_norms(self):
        def _calc_residual(theta, u, a_u):
            return a_u - theta * u

        self._residuals = map(
            _calc_residual,
            self._ritz_vals, self._ritz_vecs, self._col_ritz_vecs
        )
        self._residual_norms = map(scipy.linalg.norm, self._residuals)
        return self._residual_norms

    def _calc_trial_vecs(self):
        if self._precondition is None:
            dim = len(self._search_space[0])
            self.diag = np.zeros(dim)
            for i in range(dim):
                _v = np.zeros(dim)
                _v[i] = 1.
                self.diag[i] = self._matvec(_v)[i]

            def _precondition(residual, ritz_val, ritz_vec, noise=1.e-4):
                return residual / (ritz_val - self.diag + noise)

            self._precondition = _precondition

        self._trial_vecs = []
        precondition = self._precondition
        for i, norm in enumerate(self._residual_norms):
            # remove linear dependency in self._residuals
            if norm ** 2 > self.lin_dep_lim:
                vec = precondition(
                    self._residuals[i], self._ritz_vals[0], self._ritz_vecs[i]
                )
                vec *= 1. / scipy.linalg.norm(vec)
                for base in self._search_space:
                    vec -= np.dot(np.conj(base), vec) * base
                norm_ = scipy.linalg.norm(vec)
                # remove linear dependency between trial_vecs and
                # self._search_space
                if norm_ ** 2 > self.lin_dep_lim:
                    vec *= 1. / norm
                    self._trial_vecs.append(vec)
        return self._trial_vecs
