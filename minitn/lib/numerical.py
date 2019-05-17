#!/usr/bin/env python
# coding: utf-8
"""Numerical objects and methods.
"""
from __future__ import absolute_import, division

import logging
import math
from builtins import map, range, zip

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, norm, orth, svd
from scipy.sparse.linalg import LinearOperator

from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import time_this, unzip


class WindowFunction(object):
    """Some window functions for fourier transformation.

    References
    ----------
    [1] http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/intro_MCTDH.pdf
    """
    @staticmethod
    def g0prime(length):
        def _g(t):
            return 1. - np.abs(t) / length
        return _g

    @staticmethod
    def g1prime(length):
        def _g(t):
            y1 = (1. - np.abs(t) / length) * np.cos(np.pi * t / length)
            y2 = np.sin(np.pi * np.abs(t) / length) / np.pi
            return y1 + y2
        return _g


class BasisFunction(object):
    """Some Basis Functions.
    """
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
        from minitn.lib.symbolic import BasisFunction, lambdify
        psi = BasisFunction.harmonic_oscillator(
            n, k=k, m=m, hbar=hbar
        )
        psi = lambdify(psi)
        return psi


class PotentialFunction(object):
    """Some Potential Functions.
    """
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
    """Numerical quadrature.

    Parameters
    ----------
    func : float -> float
    start : float
    stop : float
    num_prec : int
    """
    # x = np.linspace(start, stop, num=num_prec + 1)
    # fx = func(x)
    # delta = (stop - start) / num_prec
    # area = (fx[:-1] + fx[1:]) * delta / 2.
    # quad = np.sum(area)
    return quad(func, start, stop, limit=num_prec)[0]


class DavidsonAlgorithm(object):
    """Davidson algorithm.

    Parameters
    ----------
    matvec : (N,) ndarray -> (N,) ndarray
    init_vecs : [(N,) ndarray]
        list containing k initial vectors
    n_vals : int
        Number of eigenvalues to be calculate (sorted from small to large).
    """
    tol = 1.e-12
    max_cycle = 99
    max_space = 10
    lin_dep_lim = 1.e-14
    _debug = logging.root.isEnabledFor(logging.DEBUG)

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
                logging.warning(__('No configuration "{}"!', key))
        return

    def __init__(self, matvec, init_vecs, n_vals=1, precondition=None):
        self._matvec = matvec
        self._n_vals = n_vals
        self._precondition = precondition
        self._diag = None
        self._trial_vecs = list(init_vecs)
        self._search_space = []
        self._column_space = []
        self._max_space = self.max_space + 3 * n_vals
        self._submatrix = np.zeros([self._max_space] * 2, dtype='d')

        self._last_ritz_vals = None
        self._last_convergence = None

        self._ritz_vals = None
        self._get_ritz_vecs = None    # function returns an iterator
        self._get_col_ritz_vecs = None    # function returns an iterator
        self._residuals = None    # iterator
        self._residual_norms = None
        self._convergence = None

        self.eigvals = None
        self.eigvecs = None

    def _restart(self):
        logging.debug('Search space too large, restart.')
        logging.debug(__('ritz vals: {}', self._ritz_vals))
        ritz_vals = self._ritz_vals
        self.__init__(
            matvec=self._matvec, init_vecs=self._get_ritz_vecs(),
            n_vals=self._n_vals
        )
        self._ritz_vals = ritz_vals
        return

    @time_this
    def kernel(self, search_mode=False):
        """Run Davidson algorithm.

        Returns
        -------
        eigvals : (self.n_vals,) ndarray
        eigvecs : [(n,) ndarray]
        """
        for _ in range(self.max_cycle):
            self._orthonormalize(use_svd=(not search_mode))
            self._extend_space()
            if search_mode and len(self._search_space) >= self._n_vals:
                break
            self._calc_ritz_pairs()
            self._calc_residual_norms()
            if self._is_converged():
                break
            next_ = len(self._ritz_vals) + len(self._search_space)
            if next_ > self._max_space:
                self._restart()
            else:
                self._calc_trial_vecs()

        if not search_mode:
            self.eigvals = self._ritz_vals
            self.eigvecs = list(self._get_ritz_vecs())
            return self.eigvals, self.eigvecs
        else:
            return self._search_space

    def _is_converged(self):
        if (
            self._last_ritz_vals is None or
            len(self._last_ritz_vals) != len(self._ritz_vals)
        ):
            self._convergence = [False] * len(self._ritz_vals)
            return False

        self._last_convergence = self._convergence
        diff_ritz_vals = np.abs(self._last_ritz_vals - self._ritz_vals)
        self._convergence = [
            norm_ ** 2 < self.tol and diff_ritz_vals[i] < self.tol
            for i, norm_ in enumerate(self._residual_norms)
        ]
        if self._debug:
            for _i, _norm in enumerate(self._residual_norms):
                if self._convergence[_i] and not self._last_convergence[_i]:
                    logging.debug(
                        __('Root {} converged, norm = {:.8f}', _i, _norm)
                    )
        if all(self._convergence):
            return True
        else:
            return False

    def _orthonormalize(self, use_svd=True):
        if use_svd and self._trial_vecs:
            trial_mat = np.transpose(np.array(self._trial_vecs))
            trial_mat = orth(trial_mat)
            self._trial_vecs = list(np.transpose(trial_mat))
        elif self._trial_vecs:
            vecs = []
            for vec_i in self._trial_vecs:
                for vec_j in vecs:
                    vec_i -= vec_j * np.dot(vec_j.conj(), vec_i)
                norm_ = norm(vec_i)
                if norm_ > 1.e-7:
                    vecs.append(vec_i / norm_)
            self._trial_vecs = vecs
        return self._trial_vecs

    def _extend_space(self):
        head = len(self._search_space)
        self._search_space += self._trial_vecs
        self._column_space += list(map(self._matvec, self._trial_vecs))
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
        self._ritz_vals, v = eigh(
            self._submatrix[:n_space, :n_space], eigvals=(0, n_state - 1)
        )
        v = np.transpose(v)
        self._get_ritz_vecs = (
            lambda: (_trans_vec(v_i, self._search_space) for v_i in v)
        )
        self._get_col_ritz_vecs = (
            lambda: (_trans_vec(v_i, self._column_space) for v_i in v)
        )
        return self._ritz_vals, self._get_ritz_vecs

    def _calc_residual_norms(self):
        def _calc_residual_norm(theta, u, a_u):
            residual = a_u - theta * u
            norm_ = norm(residual)
            return residual, norm_

        self._residuals, norms = unzip(map(
            _calc_residual_norm,
            self._ritz_vals, self._get_ritz_vecs(), self._get_col_ritz_vecs()
        ))
        self._residual_norms = list(norms)
        return self._residuals, self._residual_norms

    def _calc_trial_vecs(self):
        if self._precondition is None:
            dim = len(self._search_space[0])
            self._precondition = self.davidson_precondition(
                dim, self._matvec
            )
        #     ritz_vecs = [None] * dim
        # else:
        #     ritz_vecs = list(self._get_ritz_vecs())

        self._trial_vecs = []
        precondition = self._precondition
        zipped = zip(
            self._residuals, self._residual_norms,
            self._get_ritz_vecs(), self._convergence
        )
        for residual, norm_, ritz_vec, conv in zipped:
            # remove linear dependency in self._residuals
            if norm_ ** 2 > self.lin_dep_lim and not conv:
                vec = precondition(
                    residual, self._ritz_vals[0], ritz_vec
                )
                vec *= 1. / norm(vec)
                for base in self._search_space:
                    vec -= np.dot(np.conj(base), vec) * base
                norm_ = norm(vec)
                # remove linear dependency between trial_vecs and
                # self._search_space
                if norm_ ** 2 > self.lin_dep_lim:
                    vec *= 1. / norm_
                    self._trial_vecs.append(vec)
        return self._trial_vecs

    @classmethod
    def davidson_precondition(cls, dim, matvec, noise=None):
        """Stadard precondition in Davidson algorithm

        Parameters
        ----------
        dim : int
        matvec : (dim,) ndarray -> (dim,) ndarray
        noise: float, optional
        """
        if noise is None:
            noise = math.sqrt(cls.tol)
        diag = np.zeros(dim)
        for i in range(dim):
            _v = np.zeros(dim)
            _v[i] = 1.
            diag[i] = matvec(_v)[i]

        def _precondition(residual, ritz_val, ritz_vec, _diag=diag):
            return residual / (ritz_val - _diag + noise)
        return _precondition


def expection(op, vec):
    r"""Calculate the energy expection.
    Parameters
    ----------
    op : (n, n) ndarray or LinearOpearator
    vec : (n,) ndarray

    Returns
    -------
    expection : float
    """
    vec_h = np.conjugate(vec)
    e = np.dot(vec_h, op.dot(vec))
    return e


class Id(object):
    def __call__(self, vec):
        return vec

    def dot(self, vec):
        return vec


def compressed_svd(a, rank=None, err=None, **kwargs):
    u, s, vh = svd(a, full_matrices=False, **kwargs)
    if err is not None:
        total_error = 0.0
        for n, s_i in reversed(list(enumerate(s))):
            total_error += s_i
            if total_error > err:
                rank = n + 1
                break
        if rank is None:
            rank = 1
    if rank is not None and rank <= 0 :
        raise RuntimeError('The matrix must have positive rank!')
    if rank is not None and rank <= len(s):
        s = s[:rank]
        u = u[:, :rank]
        vh = vh[:rank,:]
    s = np.diag(s)
    return u, s, vh
