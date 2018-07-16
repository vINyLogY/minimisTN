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

    def solve(self, n_state=15):
        e, phi = scipy.linalg.eigh(self.h_mat(), eigvals=(0, n_state - 1))
        return e, phi

    def fbr2cont(self, vec):
        def _psi(x):
            psi = 0.0
            for j in range(self.n):
                fbr_j = self.fbr_func(j)
                psi += fbr_j(x) * vec[j]
            return psi
        return _psi

    def dvr_func(self, alpha):
        return fbr2cont(_u_mat[:, alpha])


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
        bf = BasisFunction()
        args = (i+1, self.length, self.a)
        func = bf.particle_in_box(i+1, self.length, self.a)
        return func


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


def main():
    dvr = SineDVR(-2.0, 2.0, 20)
    vf = PotentialFunction()
    v_func = vf.w_well(d0=20., a=1.)
    dvr.set_v_func(v_func)
    e, v = dvr.solve(n_state=10)
    func_set = []
    for i, e_i in enumerate(e):
        print('e{}: {}'.format(i, e_i))
        tmp = dvr.dvr2fbr_vec(v[:, i])
        tmp = dvr.fbr2cont(tmp)
        func_set.append(tmp)

    x = np.linspace(-2.0, 2.0, 500)

    n_plot = 8
    plt.figure()
    plt.subplots_adjust(left=0.05, right=0.95,
                        bottom=0.05, top=0.95)
    plt.plot(x, v_func(x), 'k-', lw=2)
    for i in range(n_plot):
        plt.plot([x[0], x[-1]], [e[i], e[i]], '--', color='gray')
    for i in range(n_plot):
        phi = np.array([func_set[i](x_) for x_ in x])
        plt.plot(x, 2. * phi + e[i])
    plt.xlim(-3., 3.)
    plt.ylim(-e[0], e[-1])
    plt.show()


if __name__ == '__main__':
    main()
