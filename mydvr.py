#!/usr/bin/env python2
# coding: utf-8

r"""# A Simple DVR Program (1-D)

## Reference

1. http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""

from __future__ import division

import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class DVR(object):
    """Vector:
        DVR <--> FBR <--> Cont.
    """

    def set_v_func(self, v_func):
        self.v_func = v_func
        return

    def v_mat(self):
        """Return the potential matrix with the given potential.
        """
        v = self.v_func(self.grid_points)
        v_matrix = np.diag(v)
        return v_matrix

    def h_mat(self):
        """Return the potential matrix with the given potential.
        """
        return self.t_mat() + self.v_mat()

    def solve(self, n_state=15):
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

    def fbr2cont(self, vec, x=None):
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
        for i in range(n_plot):
            e = self.energy[i]
            plt.plot([x[0], x[-1]], [e, e], '--', color='gray')
            phi = self.dvr2cont(self.eigenstates[i])
            phi = np.array([phi(x_) for x_ in x])
            plt.plot(x, scale * phi + e)
        y_min = min(vx)
        y_max = e + scale * max(phi)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min - scale, 1.05 * y_max)
        plt.show()
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
            chi = self.dvr_func(i)
            chi = np.array([chi(x_) for x_ in x])
            y_max = max(y_max, max(chi))
            y_min = min(y_min, min(chi))
            plt.plot(x, chi)
        plt.plot([x_min, x_max], [0., 0.], 'k-')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min * 1.05, y_max * 1.05)
        plt.show()
        return


class SineDVR(DVR):
    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        r"""From a to b.
        C. f. reference [1] 2.3.5, p30.
        """
        self.a = lower_bound
        self.b = upper_bound
        self.n = n_dvr
        self.hbar = hbar
        self.m_e = m_e
        self.length = float(self.b - self.a)
        self._calc_grid_points()
        self._calc_u_mat()

    def _calc_grid_points(self):
        step_length = self.length / (self.n + 1)
        self.grid_points = np.array(
            [self.a + step_length * i for i in range(1, self.n + 1)])
        return self.grid_points

    def _calc_u_mat(self):
        j = np.arange(1, self.n + 1)[:, None]
        a = np.arange(1, self.n + 1)[None, :]
        self._u_mat = np.sqrt(2 / (self.n + 1)) \
            * np.sin(j * a * np.pi / (self.n + 1))
        return self._u_mat

    def t_mat(self):
        """Return the kinetic energy matrix.
        """
        factor = self.hbar ** 2 / (2 * self.m_e)
        j = np.arange(1, self.n + 1)
        t_matrix = np.diag((j * np.pi / self.length)**2)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def fbr_func(self, i):
        """Return i-th FBR basis function.
        """
        bf = BasisFunction()
        args = (i+1, self.length, self.a)
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


class BasisFunction(object):
    def particle_in_box(self, j, L, x0):
        def _phi(x):
            if x0 < x and x < x0 + L:
                return np.sqrt(2. / L) * np.sin(j * np.pi * (x - x0) / L)
            else:
                return 0.
        return _phi


class PotentialFunction(object):
    def square_well(self, depth=1., width=1., x0=0., v0=0.):
        r"""Returns a function of a single variable V(x).

            (x0, v0+depth)    (x0+width, v0+depth)
                     ----+    +----
                         |    |
                (x0, v0) +----+ (x0+width, v0)
        """
        def _v(x):
            if x0 < x and x < x0 + width:
                return v0
            else:
                return v0 + depth

        return _v

    def w_well(self, d0=5., a=1.):
        r"""
                \ (0,d1) /
                 \  /\  /
            (-a,0)\/  \/(a, 0)
        """
        return lambda x: (d0 / a**4) * (x**2 - a**2)**2

    def sho(self, k=1., x0=0.):
        """Return a one-dimensional harmonic oscillator potential V(x)
        with wavenumber k.
        """
        return lambda x: 0.5 * k * (x - x0)**2


def main():
    dvr = SineDVR(-2.0, 2.0, 20)
    vf = PotentialFunction()
    v_func = vf.w_well(d0=20., a=1.)
    dvr.set_v_func(v_func)
    e, v = dvr.solve(n_state=10)
    for i, e_i in enumerate(e):
        print('e{}: {}'.format(i, e_i))
    dvr.plot_eigen(npts=100)
    dvr.plot_dvr(indices=[9, 10], npts=1000)


if __name__ == '__main__':
    main()
