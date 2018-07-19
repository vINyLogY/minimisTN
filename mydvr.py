#!/usr/bin/env python2
# coding: utf-8
r"""# A Simple DVR Program (1-D)

## Reference

1. http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""

from __future__ import division

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import sympy as sym

import mycas as cas

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class DVR(object):
    """Vector:
        DVR <--> FBR <--> Cont.

     ## Args:
    - fbr_basis: a list of sympy basis (functions).
    - trans_func_pair: (f, f^{-1}) where f(x) is monotic
        s. t. Q = <i|f(x)|j> is a tri-diagonal matrix
    """

    def __init__(self, fbr_basis, cut_off=None, trans_func_pair=(None, None),
                 cas=True, num_prec=None, hbar=1., m_e=1.):
        self.method = 'Diagonlisation_DVR'
        self.n = len(fbr_basis)
        self.basis = fbr_basis
        self.cut_off = cut_off
        self.trans_func_pair = trans_func_pair
        self.cas = cas
        self.num_prec = num_prec
        self.hbar = hbar
        self.m_e = m_e
        self._calculate_dvr()

    def _sym_calc_q_mat(self):
        """Calculate grid points and U matrix
        s. t. Q = U X U^{\dagger},
        where X_{ij} = x_i \delta_{ij},
        Q = <i|f(x)|j> is a tri-diagonal matrix
        """
        f = self.trans_func_pair[0]
        if f is None:
            f = cas.id_op()
        x = cas.x
        op = cas.prod_op(f(x))
        basis = self.basis
        Q = cas.matrix_repr(
            op, basis, cut_off=self.cut_off, num_prec=self.num_prec)
        return Q

    def _sym_calc_grid_points(self, x_i):
        inv = self.trans_func_pair[-1]
        if inv is None:
            inv = cas.id_op()
        inv = cas.lambdify(inv)
        return inv(x_i)

    def _calculate_dvr(self):
        if self.cas:
            Q = self._sym_calc_q_mat()
        x_i, self._u_mat = scipy.linalg.eigh(Q)
        if self.cas:
            self.grid_points = self._sym_calc_grid_points(x_i)
        return self.grid_points, self._u_mat

    def set_v_func(self, v_func):
        self.v_func = v_func
        return

    def v_mat(self):
        """Return the potential matrix with the given potential.
        """
        v = self.v_func(self.grid_points)
        v_matrix = np.diag(v)
        return v_matrix

    def t_mat(self):
        """Return the kinetic energy matrix.
        """
        factor = - self.hbar ** 2 / (2 * self.m_e)
        if self.cas:
            op = cas.diff(2)
            t_matrix = cas.matrix_repr(
                op, self.basis, cut_off=self.cut_off, num_prec=self.num_prec)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def h_mat(self):
        """Return the potential matrix with the given potential.
        """
        return self.t_mat() + self.v_mat()

    def solve(self, n_state=None):
        if n_state is None:
            n_state = self.n
        self.energy, v = scipy.linalg.eigh(
            self.h_mat(), eigvals=(0, n_state - 1))
        self.eigenstates = np.transpose(v)
        return self.energy, self.eigenstates

    def dvr2fbr_mat(self, mat):
        """Transform a matrix from the discrete variable representation
        to the finite basis representation.
        """
        return np.dot(self._u_mat, np.dot(mat, np.transpose(self._u_mat)))

    def dvr2fbr_vec(self, vec):
        """Transform a vector from the discrete variable representation
        to the finite basis representation.
        """
        vec = np.reshape(vec, -1)
        return np.dot(self._u_mat, vec)

    def dvr2cont(self, vec):
        """Transform a vector from the discrete variable representation.
        to the spatial function.
        """
        vec = self.dvr2fbr_vec(vec)
        psi = self.fbr2cont(vec)
        return psi

    def dvr_func(self, alpha):
        """Return alpha-th DVR basis function.
        """
        func = self.fbr2cont(self._u_mat[:, alpha])
        return func

    def fbr2dvr_mat(self, mat):
        """Transform a matrix from the finite basis representation
        to the discrete variable representation.
        """
        return np.dot(np.transpose(self._u_mat), np.dot(mat, self._u_mat))

    def fbr2dvr_vec(self, vec):
        """Transform a vector from the finite basis representation
        to the discrete variable representation.
        """
        vec = np.reshape(vec, -1)
        return np.dot(np.transpose(self._u_mat), vec)

    def fbr2cont(self, vec):
        """Transform a vector from the finite basis representation
        to the spatial function.
        """
        def _psi(x):
            psi = 0.0
            for j in range(self.n):
                fbr_j = self.fbr_func(j)
                psi += fbr_j(x) * vec[j]
            return psi
        return _psi

    def fbr_func(self, i):
        """Return i-th FBR basis function.
        """
        if self.cas:
            func = self.basis[i]
            func = cas.lambdify(func)
        return func

    def plot_eigen(self, x_min, x_max, npts=None, n_plot=None, scale=2.):
        if npts is None:
            npts = self.n
        if n_plot is None:
            n_plot = len(self.energy)
        x = np.linspace(x_min, x_max, npts)
        vx = [self.v_func(x_) for x_ in x]
        plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.plot(x, vx, 'k-', lw=2)
        y_min = min(vx)
        y_max = self.v_func(0)
        for i in range(n_plot):
            e = self.energy[i]
            plt.plot([x[0], x[-1]], [e, e], '--', color='gray')
            phi = (self.dvr2cont(self.eigenstates[i]))(x)
            plt.plot(x, scale * phi + e)
            y_max = max(y_max, e + scale * max(phi))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min - scale, 1.05 * y_max)
        plt.savefig('eigenstates-{}.png'.format(
            self.method))
        return

    def plot_dvr(self, x_min, x_max, npts=None, indices=None):
        if npts is None:
            npts = self.n
        if indices is None:
            indices = np.arange(self.n)
        x = np.linspace(x_min, x_max, npts)
        y_min = 0.
        y_max = 0.
        plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        for i in indices:
            x_i = self.grid_points[i]
            plt.plot([x_i, x_i], [-10., 10.], '--', color='gray')
            chi = (self.dvr_func(i))(x)
            if (self.dvr_func(i))(x_i) < 0.:
                chi = -1. * chi
            y_max = max(y_max, max(chi))
            y_min = min(y_min, min(chi))
            plt.plot(x, chi)
        plt.plot([x_min, x_max], [0., 0.], 'k-')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min * 1.05, y_max * 1.05)
        plt.savefig('dvr_functions-{}.png'.format(
            self.method))
        return


class SineDVR(DVR):
    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        r"""From a to b.
        C. f. reference [1] 2.3.5, p30.
        """
        self.method = 'Sine-DVR'
        self.a = lower_bound
        self.b = upper_bound
        self.n = n_dvr
        self.hbar = hbar
        self.m_e = m_e
        self.length = float(self.b - self.a)
        self._calculate_dvr()

    def _calculate_dvr(self):
        # calculate grid points
        step_length = self.length / (self.n + 1)
        self.grid_points = np.array(
            [self.a + step_length * i for i in range(1, self.n + 1)])
        # calculate U matrix
        j = np.arange(1, self.n + 1)[:, None]
        a = np.arange(1, self.n + 1)[None, :]
        self._u_mat = np.sqrt(2 / (self.n + 1)) \
            * np.sin(j * a * np.pi / (self.n + 1))
        return self.grid_points, self._u_mat

    def t_mat(self):
        """Return the kinetic energy matrix.
        """
        factor = - self.hbar ** 2 / (2 * self.m_e)
        j = np.arange(1, self.n + 1)
        t_matrix = np.diag(- (j * np.pi / self.length) ** 2)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def fbr_func(self, i):
        """Return i-th FBR basis function.
        """
        bf = cas.BasisFunction()
        func = bf.particle_in_box(i+1, self.length, self.a)
        return func

    def plot_eigen(self, x_min=None, x_max=None,
                   npts=None, n_plot=None, scale=2.):
        if x_min is None:
            min_ = self.a
        else:
            min_ = x_min
        if x_max is None:
            max_ = self.b
        else:
            max_ = x_max
        super(SineDVR, self).plot_eigen(
            min_, max_, npts=npts, n_plot=n_plot, scale=scale)
        return

    def plot_dvr(self, x_min=None, x_max=None, npts=None, indices=None):
        if x_min is None:
            min_ = self.a
        else:
            min_ = x_min
        if x_max is None:
            max_ = self.b
        else:
            max_ = x_max
        super(SineDVR, self).plot_dvr(
            min_, max_, npts=npts, indices=indices)
        return


def test_sine_dvr(x0, L, n, v_func):
    sine_dvr = SineDVR(x0, x0 + L, n)
    sine_dvr.set_v_func(v_func)
    e, v = sine_dvr.solve(n_state=5)
    for i, e_i in enumerate(e):
        print('e{}: {}'.format(i, e_i))
    sine_dvr.plot_eigen(npts=100)
    # sine_dvr.plot_dvr(npts=100)
    return


def test_dvr(x0, L, n, v_func):
    def f(x): return sym.cos(sym.pi * (x - x0) / L)

    def inv_f(y): return x0 + sym.acos(y) * L / sym.pi

    basis = [cas.particle_in_box(i, L, x0)
             for i in range(1, 1 + n)]
    dvr = DVR(basis, trans_func_pair=(f, inv_f),
              cut_off=(x0, x0 + L), num_prec=100)
    dvr.set_v_func(v_func)
    e, v = dvr.solve(n_state=5)
    for i, e_i in enumerate(e):
        print('e{}: {}'.format(i, e_i))
    dvr.plot_eigen(x0, x0 + L, npts=100)
    # dvr.plot_dvr(x0, x0 + L, npts=100)
    return


def test_improper_dvr(x0, L, n, v_func):
    basis = [cas.harmonic_oscillator(i)
             for i in range(0, n)]
    dvr = DVR(basis, cut_off=(x0, x0 + L), num_prec=100)
    dvr.set_v_func(v_func)
    dvr.method = 'improper'
    e, v = dvr.solve()
    for i, e_i in enumerate(e):
        print('e{}: {}'.format(i, e_i))
    dvr.plot_eigen(x0, x0 + L, npts=100)
    dvr.plot_dvr(x0, x0 + L, npts=100)
    return


def main():
    x0, L, n = (-5., 10., 10)
    v_func = cas.PotentialFunction().sho()
    # print('Sine-DVR:')
    # test_sine_dvr(x0, L, n, v_func)
    # print('Diagonalisation-DVR:')
    # test_dvr(x0, L, n, v_func)
    print('(Improper) Diagonalisation-DVR:')
    test_improper_dvr(x0, L, n, v_func)


if __name__ == '__main__':
    main()
