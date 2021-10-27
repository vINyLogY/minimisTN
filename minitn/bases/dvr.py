#!/usr/bin/env python2
# coding: utf-8
r"""A Simple DVR Program (n-D)

References
----------
.. [1] http://www.pci.uni-heidelberg.de/tc/usr/mctdh/lit/NumericalMethods.pdf
"""
from __future__ import absolute_import, division

import logging
import time
from builtins import filter, map, range, zip

import matplotlib.pyplot as plt

from minitn.lib.backend import np
import scipy
from scipy import fftpack
from scipy.integrate import RK45
from scipy.sparse.linalg import LinearOperator, eigsh

from minitn.lib import numerical, symbolic
from minitn.lib.numerical import DavidsonAlgorithm, expection
from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import figure


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
        Value of :math:`m_\mathrm{e}`; default is ``1.0``.

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

    def __init__(self, basis=None, grid_points=None, t_mat=None, u_mat=None, hbar=1., m_e=1.):
        r"""Some args could be ``None`` when initialized, but need to be
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
        r"""Set the potential energy function.

        Parameters
        ----------
        v_func : float -> float
        """
        self.v_func = v_func
        self._h_mat = None
        return

    def v_mat(self):
        r"""Return the potential energy matrix with the given potential
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
        r"""Return the kinetic energy matrix in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d matrix.
        """
        return self._t_mat

    def h_mat(self):
        r"""Return the Hamiltonian energy matrix in DVR.

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
        tmp = np.transpose(v)
        es = []
        for i in tmp:
            vi = i if i[0] >= 0.0 else -i
            es.append(vi)
        self.eigenstates = np.array(es)
        return self.energy, self.eigenstates

    def dvr2fbr_mat(self, mat):
        r"""Transform a matrix from DVR to FBR.

        Parameters
        ----------
        mat : (n, n) ndarray

        Returns
        -------
        (n, n) ndarray
        """
        return np.dot(self._u_mat, np.dot(mat, np.transpose(self._u_mat)))

    def dvr2fbr_vec(self, vec):
        r"""Transform a vector from DVR to FBR.

        Parameters
        ----------
        mat : (n,) ndarray

        Returns
        -------
        (n,) ndarray
        """
        return np.dot(self._u_mat, vec)

    def dvr2cont(self, vec):
        r"""Transform a vector from DVR to the spatial function.

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
        r"""Return ``alpha``-th DVR basis function.

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
        r"""Transform a matrix from FBR to DVR.

        Parameters
        ----------
        mat : (n, n) ndarray

        Returns
        -------
        (n, n) ndarray
        """
        return np.dot(np.transpose(self._u_mat), np.dot(mat, self._u_mat))

    def fbr2dvr_vec(self, vec):
        r"""Transform a vector from FBR to DVR.

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
        r"""Transform a vector from FBR to the spatial function.

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
        r"""Return ``i``-th FBR basis function.

        Parameters
        ----------
        i : int

        Returns
        -------
            ``i``-th FBR basis function.
        """
        return self.basis[i]

    def energy_expection(self, vec):
        r"""Calculate the energy expection.

        Parameters
        ----------
        vec : (n,) ndarray
            In DVR.

        Returns
        -------
        energy : float
        """
        return expection(self._h_mat, vec)

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
        r"""Plot the eigenstate together with energy and potential curve.

        Parameters
        ----------
        x_min : float
        x_max : float
        npts : int, optional
            Number of points to calculate on a single state.
        n_plot :  int, optional
            Number of states to calculate.
        scale : float, optional
            Adjust the scale of wavefunction.
        """
        if npts is None:
            npts = self.n
        if n_plot is None:
            n_plot = len(self.energy)
        x = np.linspace(x_min, x_max, npts)
        vx = [self.v_func(x_) for x_ in x]
        with figure() as fig:
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
            plt.savefig('eigenstates-{}.pdf'.format(self.comment))
        return

    def plot_func(self, func_list, x_min, x_max, y_min=0., y_max=0., npts=None):
        r"""Plot functions.

        Parameters
        ----------
        func_list : [float->float]
            List of functions to plot.
        x_min : float
        x_max : float
        y_min : float, optional
        y_max : float, optional
        npts : int, optional
            Number of points to calculate on a single state.
        """
        if npts is None:
            npts = self.n
        x = np.linspace(x_min, x_max, npts)
        with figure() as fig:
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
            for func in func_list:
                chi = func(x)
                y_max = max(y_max, max(chi))
                y_min = min(y_min, min(chi))
                plt.plot(x, chi)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min * 1.05, y_max * 1.05)
            plt.savefig('functions-{}.pdf'.format(self.comment))
        return

    def plot_dvr(self, x_min, x_max, npts=None, indices=None):
        r"""Plot DVR basis.

        Parameters
        ----------
        x_min : float
        x_max : float
        npts : int, optional
            Number of points to calculate on a single state.
        indices : (int), optional
            An iterable object containing the indices of DVR to be plotted.
        """
        if npts is None:
            npts = self.n
        if indices is None:
            indices = np.arange(self.n)
        x = np.linspace(x_min, x_max, npts)
        y_min = 0.
        y_max = 0.
        with figure() as fig:
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
            plt.savefig('dvr_functions-{}.pdf'.format(self.comment))
        return


class CasDVR(DVR):
    r"""1-d discrete variable representation using sympy to calculate
    :math:`Q = \langle i | f(x) | j \rangle`.

    Parameters
    ----------
    basis : [symbol -> symbol]
        A list of basis (FBR).
    cut_off : (float, float), optional
        Interval of the integration.
    trans_func_pair : (symbol -> symbol, symbol -> symbol)
        :math:`(f, f^{-1})` s. t. :math:`Q = \langle i | f(x) | j \rangle`
        is a tri-diagonal matrix, where :math:`f` is a monotic function.
    num_prec : int, optional
        Numerical precision for quadrature. Use analytical method if ``0``.
    hbar : float, optional
        Value of :math:`\hbar`; default is ``1.0``.
    m_e : float, optional
        Value of :math:`m_\mathrm{e}`; default is ``1.0``.

    Attributes
    ----------
    basis
    cut_off
    trans_func_pair
    num_prec
    grid_points : [float]
        a list of grid points.
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

    def __init__(self, basis, cut_off=None, trans_func_pair=(None, None), num_prec=None, hbar=1., m_e=1.):
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
        Q = symbolic.matrix_repr(op, self.basis, cut_off=self.cut_off, num_prec=self.num_prec)
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
        r"""Return the kinetic energy matrix in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d matrix.
        """
        factor = -self.hbar**2 / (2 * self.m_e)
        op = symbolic.diff(2)
        t_matrix = symbolic.matrix_repr(op, self.basis, cut_off=self.cut_off, num_prec=self.num_prec)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def fbr_func(self, i):
        r"""Return ``i``-th FBR basis function.

        Parameters
        ----------
        i : int

        Returns
        -------
        func : float -> float
        """
        func = self.basis[i]
        func = symbolic.lambdify(func)
        return func


class SineDVR(DVR):
    r"""
    Parameters
    ----------
    lower_bound : float
    upper_bound : float
    n_dvr : int
        Number of basis/grid points.
    hbar : float, optional
        Value of :math:`\hbar`; default is ``1.0``.
    m_e : float, optional
        Value of :math:`m_\mathrm{e}`; default is ``1.0``.

    Attributes
    ----------
    grid_points : [float]
        a list of grid points.
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
        class.
    """

    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        super(SineDVR, self).__init__(hbar=hbar, m_e=m_e)
        self.a = lower_bound
        self.b = upper_bound
        self.n = n_dvr
        self.length = float(self.b - self.a)
        self._calculate_dvr()

    def _calculate_dvr(self):
        # calculate grid points
        step_length = self.length / (self.n + 1)
        self.grid_points = np.array([self.a + step_length * i for i in range(1, self.n + 1)])
        # calculate U matrix
        j = np.arange(1, self.n + 1)[:, None]
        a = np.arange(1, self.n + 1)[None, :]
        self._u_mat = (np.sqrt(2 / (self.n + 1)) * np.sin(j * a * np.pi / (self.n + 1)))
        return self.grid_points, self._u_mat

    def t_mat(self):
        """Return the kinetic energy matrix in DVR.

        Returns
        -------
        (n, n) np.ndarray
            A 2-d matrix.
        """
        factor = -self.hbar**2 / (2 * self.m_e)
        j = np.arange(1, self.n + 1)
        t_matrix = np.diag(-(j * np.pi / self.length)**2)
        t_matrix = factor * self.fbr2dvr_mat(t_matrix)
        return t_matrix

    def fbr_func(self, i):
        """Return ``i``-th FBR basis function.

        Parameters
        ----------
        i : int

        Returns
        -------
        func : float -> float
        """
        func = numerical.BasisFunction.particle_in_box(i + 1, self.length, self.a)
        return func

    def plot_eigen(self, x_min=None, x_max=None, npts=None, n_plot=None, scale=2.):
        """Plot the eigenstate together with energy and potential curve.

        Parameters
        ----------
        x_min : float, optional
            Default is ``lower_bound``
        x_max : float, optional
            Default is ``upper_bound``
        npts : int, optional
            Number of points to calculate on a single state.
        n_plot :  int, optional
            Number of states to calculate.
        scale : float, optional
            Adjust the scale of wavefunction.
        """
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_eigen(min_, max_, npts=npts, n_plot=n_plot, scale=scale)
        return

    def plot_func(self, func_list, x_min=None, x_max=None, y_min=0., y_max=0., npts=None):
        """Plot functions.

        Parameters
        ----------
        func_list : [float->float]
            List of functions to plot.
        x_min : float, optional
            Default is ``lower_bound``
        x_max : float, optional
            Default is ``upper_bound``
        y_min : float, optional
        y_max : float, optional
        npts : int, optional
            Number of points to calculate on a single state.
        """
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_func(func_list, min_, max_, y_min=y_min, y_max=y_max, npts=npts)
        return

    def plot_dvr(self, x_min=None, x_max=None, npts=None, indices=None):
        """Plot DVR basis.

        Parameters
        ----------
        x_min : float
        x_max : float
        npts : int, optional
            Number of points to calculate on a single state.
        indices : (int), optional
            An iterable object containing the indices of DVR to be plotted.
        """
        min_ = self.a if x_min is None else x_min
        max_ = self.b if x_max is None else x_max
        super(SineDVR, self).plot_dvr(min_, max_, npts=npts, indices=indices)
        return


class FastSineDVR(SineDVR):
    r"""Sine DVR with ``Fast`` method for kinetic energy matrix.

    Parameters
    ----------
    lower_bound : float
    upper_bound : float
    n_dvr : int
        Number of basis/grid points.
    hbar : float, optional
        Value of :math:`\hbar`; default is ``1.0``.
    m_e : float, optional
        Value of :math:`m_\mathrm{e}`; default is ``1.0``.

    Attributes
    ----------
    grid_points : [float]
        a list of grid points.
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
        class.
    """

    def __init__(self, lower_bound, upper_bound, n_dvr, hbar=1., m_e=1.):
        # Same as SineDVR
        super(FastSineDVR, self).__init__(lower_bound, upper_bound, n_dvr, hbar=hbar, m_e=m_e)

    def h_mat(self):
        """Return the Hamiltonian energy matrix in DVR.

        Returns
        -------
        Hamitonian : LinearOperator
            A 2-d matrix.
        """

        class _Hamiltonian(LinearOperator):
            """
            Parameters
            ----------
            v_diag : (n,) ndarray
                In DVR
            t_diag : (n,) ndarray
                In FBR
            """

            def __init__(self, v_diag, t_diag):
                n = len(v_diag)
                self.v = v_diag
                self.t = t_diag / (2. * (n + 1.))
                super(_Hamiltonian, self).__init__('d', (n, n))

            def _matvec(self, vec):
                vec1 = self.v * vec
                vec2 = fftpack.dst(self.t * fftpack.dst(vec, type=1), type=1)
                return vec1 + vec2

            def _rmatvec(self, vec):
                return self._matvec(vec)

        if self._h_mat is not None:
            return self._h_mat
        v = self.v_func(self.grid_points)
        j = np.arange(1, self.n + 1)
        t = (self.hbar * j * np.pi / self.length)**2 / (2 * self.m_e)
        return _Hamiltonian(v, t)

    def propagator(self, tau=0.1, method='Trotter'):
        r"""Construct the propagator

        Parameters
        ----------
        tau : float
            Time interval at each step.
        """
        if 'Trotter' in method:
            pass


class PO_DVR(object):
    """N-dimensional DVR using sine-DVR for 1-D.

    Parameters
    ----------
    conf_list : [(float, float, int)]
    hbar : float, optional
    hbar : float, optional
    fast : bool, optional

    Attributes
    ----------
    rank : int
    n_list : [int]
    dvr_list : [SineDVR]
    v_rst : [float] -> float
    energy : [float]
    eigenstates (m, n) ndarray
    """

    def __init__(self, conf_list, hbar=1., m_e=1., fast=False):
        self.rank = len(conf_list)
        self.n_list = []
        self.bound_list = []
        self.dvr_list = []
        DVR_1d = FastSineDVR if fast else SineDVR
        for i in range(self.rank):
            lower_bound, upper_bound, n_dvr = conf_list[i]
            self.n_list.append(n_dvr)
            self.bound_list.append((lower_bound, upper_bound))
            sp_dvr = DVR_1d(lower_bound, upper_bound, n_dvr, hbar=hbar, m_e=m_e)
            self.dvr_list.append(sp_dvr)
        self.dim = np.prod(self.n_list)
        self.grid_points_list = [dvr_i.grid_points for dvr_i in self.dvr_list]
        self.hbar = hbar

        self.h_list = None
        self.v_rst = None
        self._diag_v_rst = None
        self.energy = None
        self.eigenstates = None

    def energy_expection(self, vec):
        return expection(self.h_mat(), vec)

    def set_v_func(self, v_list, v_rst=None):
        """Set the potential energy function.

        Parameters
        ----------
        v_list : [float -> float]
            A list of 1-arg functions
        v_rst : [float] -> float
            A 1-ary function, where the arg is a list of which length is
            ``rank``.
        """
        for i, v_i in enumerate(v_list):
            self.dvr_list[i].set_v_func(v_i)
        self.v_rst = v_rst
        self.h_list = [dvr.h_mat() for dvr in self.dvr_list]
        if self.v_rst is not None:
            self._diag_v_rst = self._calc_diag(self.v_rst)
        return

    def _calc_diag(self, func):
        v = []
        for i in range(self.dim):
            x = []
            sub = self.subindex(i)
            for j, n in enumerate(sub):
                x.append(self.grid_points_list[j][n])
            v.append(func(x))
        return np.array(v)

    def h_mat(self):
        """Return the Hamiltonian energy matrix in DVR.

        Returns
        -------
        (N, N) LinearOperator
        """

        class _Hamiltonian(LinearOperator):
            """All parameters are in DVR.

            Parameters
            ----------
            h_list : [LinearOperator]
                A list of H_i matrixes
            v_rst : (N,) ndarray
                diagonal of V_rst matrix
            """

            def __init__(self, h_list, v_rst):
                self.h_list = h_list
                self.v_rst = v_rst
                self.io_sizes = [h_i.shape[0] for h_i in h_list]
                shape = [np.prod(self.io_sizes)] * 2
                super(_Hamiltonian, self).__init__('d', shape)

            def _matvec(self, vec):
                v = np.reshape(vec, self.io_sizes)
                ans = np.zeros_like(v)
                for i, h_i in enumerate(self.h_list):
                    v_i = np.swapaxes(v, -1, i)
                    size_i = (self.io_sizes[:i] + self.io_sizes[i + 1:] + [self.io_sizes[i]])
                    v_i = np.reshape(v_i, (-1, self.io_sizes[i]))
                    tmp = np.array(list(map(h_i.dot, v_i)))
                    tmp = np.reshape(tmp, size_i)
                    ans += np.swapaxes(tmp, -1, i)
                ans = np.reshape(ans, -1)
                if self.v_rst is not None:
                    ans = ans + self.v_rst * vec
                return ans

            def _rmatvec(self, vec):
                return self._matvec(vec)

        return _Hamiltonian(self.h_list, self._diag_v_rst)

    def solve(self, n_state=1, davidson=False):
        r"""Solve the TISE with the potential energy given.

        Parameters
        ----------
        n_state : int, optional
            Number of states to be calculated, sorted by energy from low to
            high.
        davidson : bool, optional
            Whether use Davidson method.

        Returns
        -------
        energy : [float]
        eigenstates : np.ndarray
        """
        h_op = self.h_mat()
        v = self.init_state()
        if davidson:
            solver = DavidsonAlgorithm(h_op.dot, [v], n_vals=n_state)
            self.energy, self.eigenstates = solver.kernel()
        else:
            self.energy, v = eigsh(h_op, k=n_state, which='SA', v0=v)
            self.eigenstates = np.transpose(v)
        return self.energy, self.eigenstates

    def init_state(self):
        v = 1.
        for i in range(self.rank):
            _, v_i = self.dvr_list[i].solve(n_state=1)
            v = np.tensordot(v, v_i[0], axes=0)
        v = np.reshape(v, -1)
        return v

    def mu_mat(self, dim):
        """Return the dipole moment matrix in DVR at direction ``dim``.

        Parameters
        ----------
        dim : int

        Returns
        -------
        (N, N) LinearOperator

        Notes
        -----
        Suppose dim 0, 1, 2 correspond to x, y, z.
        Ignore constant factor.
        """

        class _Mu(LinearOperator):

            def __init__(self, diag_mu):
                self._diag = diag_mu
                shape = [len(diag_mu)] * 2
                super(_Mu, self).__init__('d', shape)

            def _matvec(self, vec):
                return self._diag * vec

        diag_mu = self._calc_diag(lambda args: 1. - args[dim])
        return _Mu(diag_mu)

    def propagation(self,
                    init=None,
                    start=0.,
                    stop=5.,
                    max_inter=0.01,
                    const_energy=None,
                    updater=None,
                    normalizer=None):
        """A generator doing the propagation.

        Parameters
        ----------
        init : (N,) ndarray, optional
            Real initial vector at t = t_0.
        start : float, optional
            Time starting point t_0.
        stop : float, optional
            End point t_1.
        max_inter : float, optional
            Maximum of time gap.
        const_enengy : bool, optional
            Whether to keep energy as a constant
        updater : -> a, optional
            Action after computing one step.
        normalizer : (N,) ndarray -> (N,) ndarray

        Yields
        ------
        tuple : (float, ((N,) ndarray, (N,) ndarray))
            (time, (real_vector, imaginary_vector))
        """
        if init is None:
            init = self.init_state()
        h_op = self.h_mat()
        e0 = self.energy_expection(init)
        length = len(init)
        init = init.astype(complex)
        factor = 1.0 / self.hbar

        def propagator(t, y):
            real, imag = y.real, y.imag
            real, imag = factor * h_op(imag), -factor * h_op(real)
            y = real + 1.0j * imag
            return y

        solver = RK45(propagator, start, init, stop, max_step=max_inter)
        while True:
            # Normalization
            if normalizer is not None:
                solver.y = normalizer(solver.y)

            t = solver.t
            y = solver.y
            real, imag = y.real, y.imag
            e = self.energy_expection(y)
            logging.info(__("t: {:.3f}, E: {:.8f}, |v|^2: {:.8f}", t, e, scipy.linalg.norm(y)**2))
            if abs(e - e0) > 1.e-8:
                logging.warning('Energy is not conserved. ')
                if const_energy and abs(e - e0) > const_energy:
                    logging.warning(__('Propagation stopped at t = {:.3f}.', solver.t))
                    raise StopIteration
            yield t, (real, imag)
            try:
                solver.step()
            except RuntimeError:
                logging.debug('Iterator normal terminated.')
                raise StopIteration
            else:
                if updater is not None:
                    updater(solver)

    def plot_propagation(self, init=None, start=0., stop=1., max_inter=0.01, filt=1.e-6, sample=20):
        """Plot propagation.
        """

        def string(t):
            return '{:08d}'.format(int(t))

        if init is None:
            init = self.init_state()
        it = self.propagation(init=init, start=start, stop=stop, max_inter=max_inter)
        plotter = self.plot_wf(init, string(0))
        plotter.next()
        for i, (t, vec) in enumerate(it):
            if i % sample:
                continue
            real, _ = vec
            msg = string(1000 * t)
            plotter.send((real, msg))
        plotter.close()
        return

    def plot_wf(self, vec, msg=None):
        """Plot 2D wavefunction.

        Parameters
        ----------
        vec : (N,) ndarray
        msg : string

        Notes
        -----
        This is a generator. Use .next() to plot, and .send(vec, msg) to
        plot next wavefunction with the same figure parameters.
        """
        x_lim, y_lim = self.bound_list[:2]
        x, y = self.grid_points_list[:2]
        shape = self.n_list[:2]
        x, y = np.meshgrid(x, y)
        z_lim = int(np.max(np.abs(vec)) * 15) / 10
        bound = np.linspace(-z_lim, z_lim, 100)
        while vec is not None:
            vec = np.reshape(vec, shape)
            if msg is None:
                msg = int(time.time())
            with figure() as fig:
                plt.contourf(x, y, vec, bound, cmap='seismic')
                plt.colorbar(boundaries=bound)
                plt.savefig('functions-{}.pdf'.format(msg))
            received = yield
            if received is None:
                raise StopIteration
            else:
                vec, msg = received

    def autocorrelation(self, init=None, stop=5., max_inter=0.01, dot=None, **kwargs):
        """Time autocorrelation function generator.

        Yields
        ------
        tuple : float, complex
            (t, auto)
        """
        t_2 = stop / 2
        it = self.propagation(init=init, start=0., stop=stop / 2, max_inter=max_inter, **kwargs)
        if dot is None:
            dot = np.dot
        for i, (tau, (real, imag)) in enumerate(it):
            if tau >= t_2:
                raise StopIteration
            t = tau * 2
            auto = dot(real, real) - dot(imag, imag) + 2.0j * dot(real, imag)
            yield t, auto

    def spectrum(self, init=None, length=5., max_inter=0.01, window=None, **kwargs):
        """Power spectrum.

        Parameters
        ----------
        init : (N,) ndarray, optional
        cut : float
        max_inter : float
        window : float -> float
            Window function with length.

        Returns
        -------
        freq : [float]
        sigma : [complex]
        """
        t, auto = zip(*self.autocorrelation(init=init, stop=length, max_inter=max_inter, **kwargs))
        n = len(t)
        tau = (t[-1] - t[0]) / (n - 1)
        uniform = all(abs(t_i - i * tau) < 1.e-8 for i, t_i in enumerate(t))

        if window is not None:
            zipped = zip(t, auto)
            auto = [a_i * window(t) for t, a_i in zipped]

        if uniform:
            omega = 2 * np.pi / n / tau
            freq = np.arange(n) * omega
            sigma = fftpack.ifft(auto)
            return freq, sigma
        else:
            raise NotImplementedError("Need NUDFT, or use smaller max_inter.")

    def subindex(self, N):
        """
        Parameters
        ----------
        N : int

        Returns
        -------
        sub : [int]
        """
        prods = [1]
        for i in range(self.rank - 1, 0, -1):
            prods.append(prods[-1] * self.n_list[i])
        sub = []
        for i in range(self.rank):
            base = prods.pop()
            sub.append(N // base)
            N = N % base
        return sub
