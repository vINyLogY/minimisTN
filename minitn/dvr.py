#!/usr/bin/env python2
# coding: utf-8
r"""A Simple DVR Program (n-D)

References
----------
.. [1] http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""

from __future__ import division

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import fftpack
from scipy.sparse.linalg import LinearOperator, eigsh

from minitn.lib import numerical, symbolic
from minitn.lib.numerical import DavidsonAlgorithm


class DVR(object):
    r"""1-d discrete variable representation

    Parameters
    ----------
    basis : [a -> a]
        A list of basis (FBR).
    grid_points : [float]
        a list of grid points.
    t_mat : array_like
        The matrix representation of kinetic energy in FBR.
    u_mat : array_like
        The matrix transform a vector from DVR to FBR.
    hbar : float, optional
        Value of :math:`\hbar`; default is ``1.0``.
    m_e : float, optional
        Value of :math:`\m_e`; default is ``1.0``.

    Attributes
    ----------
    basis
    grid_points
    hbar
    m_e
    n : int
        Number of basis/grid points.
    v_func : float -> float
        1-ary function of potential energy.
    energy : [float]
        List with ``m`` elements.
    eigenstates : (m, n) ndarray
        2-d array, ``eigenstates[i]`` corresponds to ``energy[i]``.
    comment : string
        Additional string in the file name if plot. Default is the name of
        class
    """

    def __init__(self, basis=None, grid_points=None,
                 t_mat=None, u_mat=None, hbar=1., m_e=1.):
        """Some args could be ``None`` when initialized, but need to be
        specified later.

        Parameters
        ----------
        basis : [a -> a]
            A list of basis (FBR).
        grid_points : [float]
            a list of grid points.
        t_mat : (n, n) ndarray
            The matrix representation of kinetic energy in FBR.
        u_mat : (n, n) ndarray
            The matrix transform a vector from DVR to FBR.
        hbar : float, optional
            Value of :math:`\hbar`; default is ``1.0``.
        m_e : float, optional
            Value of :math:`m_\mathrm{e}`; default is ``1.0``.
        """
        self.basis = basis
        self.grid_points = grid_points
        self._u_mat = u_mat
        self._t_mat = t_mat
        self.hbar = hbar
        self.m_e = m_e

        self.n = None if basis is None else len(basis)
        self.v_func = None
        self.energy = None
        self.eigenstates = None
        self._h_mat = None
        self.comment = type(self).__name__
        return

    def set_v_func(self, v_func):
        """Set the potential energy function.

        Parameters
        ----------
        v_func : float -> float
        """
        self.v_func = v_func
        self._h_mat = None
        return

    def v_mat(self):
        """Return the potential energy matrix with the given potential
        in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d diagonal matrix.
        """
        v = self.v_func(self.grid_points)
        v_matrix = np.diag(v)
        return v_matrix

    def t_mat(self):
        """Return the kinetic energy matrix in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d matrix.
        """
        return self._t_mat

    def h_mat(self):
        """Return the Hamiltonian energy matrix in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d matrix.
        """
        if self._h_mat is not None:
            return self._h_mat
        return self.t_mat() + self.v_mat()

    def solve(self, n_state=None):
        r"""Solve the TISE with the potential energy given.

        Parameters
        ----------
        n_state : int, optional
            Number of states to be calculated, sorted by energy from low to
            high.

        Returns
        -------
        energy : [float]
        eigenstates : np.ndarray

        See Also
        ________
        DVR : Definition of all attributes.
        """
        if n_state is None:
            n_state = self.n - 1
        self._h_mat = self.h_mat()
        self.energy, v = eigsh(self._h_mat, k=n_state, which='SA')
        # self.energy, v = scipy.linalg.eigh(
        #     self._h_mat, eigvals=(0, n_state - 1))
        self.eigenstates = np.transpose(v)
        return self.energy, self.eigenstates

    def dvr2fbr_mat(self, mat):
        """Transform a matrix from DVR to FBR.

        Parameters
        ----------
        mat : (n, n) ndarray

        Returns
        -------
        (n, n) ndarray
        """
        return np.dot(self._u_mat, np.dot(mat, np.transpose(self._u_mat)))

    def dvr2fbr_vec(self, vec):
        """Transform a vector from DVR to FBR.

        Parameters
        ----------
        mat : (n,) ndarray

        Returns
        -------
        (n,) ndarray
        """
        return np.dot(self._u_mat, vec)

    def dvr2cont(self, vec):
        """Transform a vector from DVR to the spatial function.

        Parameters
        ----------
        mat : (n,) ndarray

        Returns
        -------
        float -> float
        """
        vec = self.dvr2fbr_vec(vec)
        psi = self.fbr2cont(vec)
        return psi

    def dvr_func(self, alpha):
        """Return ``alpha``-th DVR basis function.

        Parameters
        ----------
        alpha : int

        Returns
        -------
            ``alpha``-th FBR basis function.
        """
        func = self.fbr2cont(self._u_mat[:, alpha])
        return func

    def fbr2dvr_mat(self, mat):
        """Transform a matrix from FBR to DVR.

        Parameters
        ----------
        mat : (n, n) ndarray

        Returns
        -------
        (n, n) ndarray
        """
        return np.dot(np.transpose(self._u_mat), np.dot(mat, self._u_mat))

    def fbr2dvr_vec(self, vec):
        """Transform a vector from FBR to DVR.

        Parameters
        ----------
        mat : (n,) ndarray

        Returns
        -------
        (n,) ndarray
        """
        vec = np.reshape(vec, -1)
        return np.dot(np.transpose(self._u_mat), vec)

    def fbr2cont(self, vec):
        """Transform a vector from FBR to the spatial function.

        Parameters
        ----------
        mat : (n,) ndarray

        Returns
        -------
        float -> float
        """
        def _psi(x):
            psi = 0.0
            for j in range(self.n):
                fbr_j = self.fbr_func(j)
                psi += fbr_j(x) * vec[j]
            return psi
        return _psi

    def fbr_func(self, i):
        """Return ``i``-th FBR basis function.

        Parameters
        ----------
        i : int

        Returns
        -------
            ``i``-th FBR basis function.
        """
        return self.basis[i]

    def energy_expection(self, vec):
        """Calculate the energy expection.

        Parameters
        ----------
        vec : (n,) ndarray
            In DVR.

        Returns
        -------
        energy : float
        """
        dim = len(vec)
        vec = np.reshape(vec, (dim, 1))
        vec_h = np.conjugate(np.transpose(vec))
        e = np.dot(vec_h, np.dot(self._h_mat, vec))
        return e[0, 0]

    def propagator(self, tau=0.1, method='Trotter'):
        r"""Construct the propagator

        Parameters
        ----------
        tau : float
            Time interval at each step.

        Returns
        -------
        p1 : (n, n) ndarray
            :math:`e^{-iV\tau/2}`
        p2 : (n, n) ndarray
            :math:`e^{-iV\tau}`
        p3 : (n, n) ndarray
            :math:`e^{-iT\tau}`
        """
        # TODO: use non-dense method.
        hbar = self.hbar
        if 'Trotter' in method:
            diag, v = scipy.linalg.eigh(self.t_mat())
            p3 = np.exp(-1.0j * hbar * tau * diag)
            p3 = np.dot(v, np.dot(np.diag(p3), np.transpose(v)))
            p2 = np.exp(-1.0j * hbar * tau * np.diag(self.v_mat()))
            p2 = np.diag(p2)
            p1 = np.exp(-1.0j * hbar * tau * np.diag(0.5 * self.v_mat()))
            p1 = np.diag(p1)
            return p1, p2, p3

    def plot_eigen(self, x_min, x_max, npts=None, n_plot=None, scale=2.):
        if npts is None:
            npts = self.n
        if n_plot is None:
            n_plot = len(self.energy)
        x = np.linspace(x_min, x_max, npts)
        vx = [self.v_func(x_) for x_ in x]
        fig = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        plt.plot(x, vx, 'k-', lw=2)
        y_min = min(vx)
        y_max = self.v_func(0)
        for i in range(n_plot):
            e = self.energy[i]
            plt.plot([x[0], x[-1]], [e, e], '--', color='gray')
            phi = (self.dvr2cont(self.eigenstates[i]))(x)
            plt.plot(x, scale * phi + e)
            y_max = min(y_min, e - scale * min(phi))
            y_max = max(y_max, e + scale * max(phi))
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, 1.05 * y_max)
        plt.savefig('eigenstates-{}.png'.format(self.comment))
        plt.close(fig)
        return

    def plot_func(self, func_list, x_min, x_max,
                  y_min=0., y_max=0., npts=None):
        if npts is None:
            npts = self.n
        x = np.linspace(x_min, x_max, npts)
        fig = plt.figure()
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
        for func in func_list:
            chi = func(x)
            y_max = max(y_max, max(chi))
            y_min = min(y_min, min(chi))
            plt.plot(x, chi)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min * 1.05, y_max * 1.05)
        plt.savefig('functions-{}.png'.format(self.comment))
        plt.close(fig)
        return

    def plot_dvr(self, x_min, x_max, npts=None, indices=None):
        if npts is None:
            npts = self.n
        if indices is None:
            indices = np.arange(self.n)
        x = np.linspace(x_min, x_max, npts)
        y_min = 0.
        y_max = 0.
        fig = plt.figure()
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
        plt.savefig('dvr_functions-{}.png'.format(self.comment))
        plt.close(fig)
        return


class CasDVR(DVR):
    r"""
    trans_func_pair: (f, f^{-1}) where f(x) is monotic
        s. t. Q = <i|f(x)|j> is a tri-diagonal matrix
    """
    def __init__(self, basis, cut_off=None, trans_func_pair=(None, None),
                 num_prec=None, hbar=1., m_e=1.):
        super(CasDVR, self).__init__(basis=basis, hbar=hbar, m_e=m_e)
        self.cut_off = cut_off
        self.trans_func_pair = trans_func_pair
        self.num_prec = num_prec
        self._calculate_dvr()

    def _sym_calc_q_mat(self):
        r"""Calculate grid points and U matrix
        s. t. Q = U X U^{\dagger},
        where X_{ij} = x_i \delta_{ij},
        Q = <i|f(x)|j> is a tri-diagonal matrix
        """
        f = self.trans_func_pair[0]
        if f is None:
            f = symbolic.id_op()
        x = symbolic.x
        op = symbolic.prod_op(f(x))
        Q = symbolic.matrix_repr(
            op, self.basis, cut_off=self.cut_off, num_prec=self.num_prec)
        return Q

    def _sym_calc_grid_points(self, x_i):
        inv = self.trans_func_pair[-1]
        if inv is None:
            inv = symbolic.id_op()
        inv = symbolic.lambdify(inv)
        return inv(x_i)

    def _calculate_dvr(self):
        Q = self._sym_calc_q_mat()
        x_i, self._u_mat = scipy.linalg.eigh(Q)
        self.grid_points = self._sym_calc_grid_points(x_i)
        return self.grid_points, self._u_mat

    def t_mat(self):
        """Return the kinetic energy matrix.
        """
        factor = - self.hbar ** 2 / (2 * self.m_e)
        op = symbolic.diff(2)
        t_matrix = symbolic.matrix_repr(
            op, self.basis, cut_off=self.cut_off, num_prec=self.num_prec)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def fbr_func(self, i):
        """Return i-th FBR basis function.
        """
        func = self.basis[i]
        func = symbolic.lambdify(func)
        return func


class SineDVR(DVR):
    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        r"""From a to b.
        C. f. reference [1] 2.3.5, p30.
        """
        super(SineDVR, self).__init__(hbar=hbar, m_e=m_e)
        self.a = lower_bound
        self.b = upper_bound
        self.n = n_dvr
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
        self._u_mat = (
            np.sqrt(2 / (self.n + 1)) * np.sin(j * a * np.pi / (self.n + 1))
        )
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
        func = numerical.BasisFunction.particle_in_box(
            i + 1, self.length, self.a
        )
        return func

    def plot_eigen(self, x_min=None, x_max=None,
                   npts=None, n_plot=None, scale=2.):
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_eigen(
            min_, max_, npts=npts, n_plot=n_plot, scale=scale)
        return

    def plot_func(self, func_list,
                  x_min=None, x_max=None, y_min=0., y_max=0., npts=None):
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_func(
            func_list, min_, max_, y_min=y_min, y_max=y_max, npts=npts)
        return

    def plot_dvr(self, x_min=None, x_max=None, npts=None, indices=None):
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_dvr(
            min_, max_, npts=npts, indices=indices)
        return


class FastSineDVR(SineDVR):
    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        """Same as SineDVR"""
        super(FastSineDVR, self).__init__(
            lower_bound, upper_bound, n_dvr, hbar=hbar, m_e=m_e)

    def h_mat(self):
        class _Hamiltonian(LinearOperator):
            def __init__(self, v_diag, t_diag):
                """
                - v_diag: in DVR
                - t_diag: in FBR
                """
                n = len(v_diag)
                self.v = v_diag
                self.t = t_diag / (2. * (n + 1.))
                super(_Hamiltonian, self).__init__('d', (n, n))

            def _matvec(self, vec):
                vec1 = self.v * vec
                vec2 = fftpack.dst(self.t * fftpack.dst(vec, type=1), type=1)
                return vec1 + vec2

            def _rmatvec(self, vec): return self._matvec(vec)

        if self._h_mat is not None:
            return self._h_mat
        v = self.v_func(self.grid_points)
        j = np.arange(1, self.n + 1)
        t = self.hbar ** 2 / (2 * self.m_e) * (j * np.pi / self.length) ** 2
        return _Hamiltonian(v, t)


class PO_DVR(object):
    """N-dimensional DVR
    """
    def __init__(self, conf_list, hbar=1., m_e=1., fast=False):
        self.rank = len(conf_list)
        self.n_list = []
        self.dvr_list = []
        DVR_1d = FastSineDVR if fast else SineDVR
        for i in range(self.rank):
            lower_bound, upper_bound, n_dvr = conf_list[i]
            self.n_list.append(n_dvr)
            sp_dvr = DVR_1d(
                    lower_bound, upper_bound, n_dvr, hbar=hbar, m_e=m_e
            )
            self.dvr_list.append(sp_dvr)
        self.dim = np.prod(self.n_list)
        self.grid_points_list = np.array(
            [dvr_i.grid_points for dvr_i in self.dvr_list]
        )

        self.v_rst = None
        self._diag_v_rst = None
        self.energy = None
        self.eigenstates = None

    def set_v_func(self, v_list, v_rst=None):
        """
        v_list: a list of 1-arg functions
        v_rst: a 1-arg function, where the arg is a list
            of which length is self.rank
        """
        for i, v_i in enumerate(v_list):
            self.dvr_list[i].set_v_func(v_i)
        self.v_rst = v_rst
        if self.v_rst is not None:
            self._calc_diag_v_rst()
        return

    def _calc_diag_v_rst(self):
        v = []
        for i in range(self.dim):
            x = []
            sub = self.subindex(i)
            for j, n in enumerate(sub):
                x.append(self.grid_points_list[j, n])
            v.append(self.v_rst(x))
        self._diag_v_rst = np.array(v)
        return self._diag_v_rst

    def h_mat(self, direct=True):
        class _Hamiltonian(LinearOperator):
            def __init__(self, h_list, v_rst):
                """All in DVR.
                - h_list: a list of H_i matrixes
                - v_rst: diagonal of V_rst matrix
                """
                self.h_list = h_list
                self.v_rst = v_rst
                self.io_sizes = [h_i.shape[0] for h_i in h_list]
                shape = [np.prod(self.io_sizes)] * 2
                super(_Hamiltonian, self).__init__('d', shape)

            def _matvec(self, vec):
                v = np.reshape(vec, self.io_sizes)
                ans = np.zeros(self.io_sizes)
                for i, h_i in enumerate(self.h_list):
                    v_i = np.swapaxes(v, -1, i)
                    size_i = (
                        self.io_sizes[:i] +
                        self.io_sizes[i + 1:] +
                        [self.io_sizes[i]]
                    )
                    v_i = np.reshape(v_i, (-1, self.io_sizes[i]))
                    tmp = np.array(map(h_i.dot, v_i))
                    tmp = np.reshape(tmp, size_i)
                    ans += np.swapaxes(tmp, -1, i)
                ans = np.reshape(ans, -1)
                if self.v_rst is not None:
                    ans = ans + self.v_rst * vec
                return ans

            def _rmatvec(self, vec): return self._matvec(vec)

        h_list = []
        for i in range(self.rank):
            h_list.append(self.dvr_list[i].h_mat())
        if direct:
            return _Hamiltonian(h_list, self._diag_v_rst)

    def solve(self, n_state=1, davidson=True):
        v = 1.
        for i in range(self.rank):
            _, v_i = self.dvr_list[i].solve(n_state=1)
            v = np.tensordot(v, v_i[0], axes=0)
        v = np.reshape(v, -1)
        h_op = self.h_mat()
        if davidson:
            solver = DavidsonAlgorithm(h_op.dot, [v], n_vals=n_state)
            self.energy, self.eigenstates = solver.kernel()
        else:
            self.energy, v = eigsh(h_op, k=n_state, which='SA', v0=v)
            self.eigenstates = np.transpose(v)
        return self.energy, self.eigenstates

    def subindex(self, N):
        prods = [1]
        for i in range(self.rank - 1, 0, -1):
            prods.append(prods[-1] * self.n_list[i])
        sub = []
        for i in range(self.rank):
            base = prods.pop()
            sub.append(N // base)
            N = N % base
        return sub
