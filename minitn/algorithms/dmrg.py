#!/usr/bin/env python2
# coding: utf-8
r"""A Simple MPS/MPO Program.

Definition
----------
- MPS matrix: a 3-index tensor A[s, i, j]::

            |s
        -i- A -j-

    [s]: local Hilbert space,
    [i,j]: virtual indices.

- MPO matrix: a 4-index tensor W[a, b, s, t]::

            |s
        -a- W -b-
            |t

    [s, t]: local Hilbert space,
    [a, b]: virtual indices.

Reference
---------
.. [1] arXiv:1008.3477v2
   [2] arXiv:1603.03039v4
   [3] https://people.smp.uq.edu.au/IanMcCulloch/mptoolkit/index.php
"""
from __future__ import absolute_import, division

import logging
import math

from minitn.lib.backend import np
from scipy.sparse.linalg import LinearOperator, eigsh


def fm_state(N, anti=False):
    r"""Get a 1-D n-site antiferromagnetic state |+-+- ... +->
    or ferromagnetic state |++ ... +>

    Parameters
    ----------
    N : int
        (Even) number of sites.
    anti : bool

    Returns
    -------
    mps : [(2, 1, 1) ndarray]
        A list of MPS matrixes.
    """
    # local bond dimension, 0=up, 1=down
    up = np.zeros((2, 1, 1))
    up[0, 0, 0] = 1
    down = np.zeros((2, 1, 1))
    down[1, 0, 0] = 1
    if anti:
        mps = [up, down] * int(N / 2)
    else:
        mps = [up] * N
    return mps


def heisenberg(N, J=1.0, Jz=1.0, h=0):
    r"""Generate a MPO for Heisenberg Model.

    .. math::

        H = \sum^{N-2}_{i=0} \frac{J}{2} (S^+_i S^-_{i+1} + S^-_i S^+_{i+1})
            + J_z S^z_i S^z_{i+1}
            - \sum^{N-1}_{i=0} h S^z_i

    For 1-D antiferromagnetic, :math:`J = J_z = 1`.

    Parameters
    ----------
    N : int
        number of sites.
    J : float
        coupling constant.
    Jz : float
        coupling constant in z-direction.
    h : float
        external magnetic field.

    Returns
    -------
    mpo : [(5, 5, 2, 2) ndarray]
        A list of MPO matrixes.
    """
    # Local operators
    I = np.eye(2)
    Z = np.zeros((2, 2))
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0., 0.], [1., 0.]])
    Sm = np.array([[0., 1.], [0., 0.]])
    # left-hand edge: 1*5
    Wfirst = np.array([[-h * Sz, (J / 2.) * Sm, (J / 2.) * Sp, (Jz / 2.) * Sz, I]])
    # mid: 5*5
    W = np.array([[I, Z, Z, Z, Z], [Sp, Z, Z, Z, Z], [Sm, Z, Z, Z, Z], [Sz, Z, Z, Z, Z],
                  [-h * Sz, (J / 2.) * Sm, (J / 2.) * Sp, (Jz / 2.) * Sz, I]])
    # right-hand edge: 5*1
    Wlast = np.array([[I], [Sp], [Sm], [Sz], [-h * Sz]])
    mpo = [Wfirst] + ([W] * (N - 2)) + [Wlast]
    return mpo


def contract_from_left(L, A, W, B):
    r"""Tensor contraction from the left hand side.::

        + -j-     + -i- A -j-
        |         |     |s
        L'-b-  =  L -a- W -b-
        |         |     |t
        + -l-     + -k- B -l-

    Parameters
    ----------
    L : (a, i, k) ndarray
    A : (s, i, j) ndarray
    W : {(a, b, s, t) ndarray, None}
    B : (t, k, l) ndarray

    Returns
    -------
    L' : {(b, j, l) ndarray, (j, l) ndarray}
        L'[b, j, l] if ``W`` is not ``None``, otherwise L'[j, l].
    """
    if W is not None:
        temp = np.einsum('sij,aik->sajk', A, L)
        temp = np.einsum('sajk,abst->tbjk', temp, W)
        temp = np.einsum('tbjk,tkl->bjl', temp, B)
    else:
        temp = np.einsum('sij,ik->sjk', A, L)
        temp = np.einsum('sjk,skl->jl', temp, B)
    return temp


def contract_from_right(R, A, W, B):
    r"""Tensor contraction from the right hand side.::

        -i- +    -i- A -j- +
            |        |s    |
        -a- R' = -a- W -b- R
            |        |t    |
        -k- +    -k- B -l- +

    Parameters
    ----------
    R : (b, j, l) ndarray
    A : (s, i, j) ndarray
    W : (a, b, s, t) ndarray
    B : (t, k, l) ndarray

    Returns
    -------
    R' : {(a, i, k) ndarray, (i, k) ndarray}
        L'[a, i, k] if ``W`` is not ``None``, otherwise L'[i, k].
    """
    if W is not None:
        temp = np.einsum('sij,bjl->sbil', A, R)
        temp = np.einsum('sbil,abst->tail', temp, W)
        temp = np.einsum('tail,tkl->aik', temp, B)
    else:
        temp = np.einsum('sij,jl->sil', A, R)
        temp = np.einsum('sil,skl->ik', temp, B)
    return temp


class EnvTensor(LinearOperator):
    r"""An environment tensor to evaluate the Hamiltonian matrix-vector
    multiply.::

        i s k   + -i-   -k- +
        | | |   |     |s    |
        + H + = L -a- W -b- R
        | | |   |     |t    |
        j t l   + -j-   -l- +

    Parameters
    ----------
    L : (a, i, j) ndarray
    W : (a, b, s, t) ndarray
    R : (b, k, l) ndarray
    """

    def __init__(self, L, W, R):
        self.L = L
        self.W = W
        self.R = R
        self.io_shape = [W.shape[3], L.shape[2], R.shape[2]]
        self.size = np.prod(self.io_shape)
        super(EnvTensor, self).__init__('d', (self.size, self.size))

    def _matvec(self, A):
        temp = np.einsum('aij,sik->ajsk', self.L, np.reshape(A, self.io_shape))
        temp = np.einsum('ajsk,abst->bjtk', temp, self.W)
        temp = np.einsum('bjtk,bkl->tjl', temp, self.R)
        return np.reshape(temp, self.size)


def opt_one_site(env_tensor, A):
    r"""Solve eigenvalue problem at one site A.

    Parameters
    ----------
    env_tensor : LinearOperator
        A linear operator, which is the environment tensor of A s. t.
        (t, j, l) ndarray -> (s, i, k) ndarray.
    A : (t, j, l) ndarray
        A MPS matrix.

    Returns
    -------
    E : float
        Rayleigh quotient
    V : (t, j, l) ndarray
        optimized one-site MPS matrix
    """

    A = np.reshape(A, env_tensor.size)
    E, V = eigsh(env_tensor, 1, v0=A, which='SA')
    E, V = E[0], np.reshape(V[:, 0], env_tensor.io_shape)
    # A = np.reshape(V, env_tensor.size)
    # E_squared = np.dot(A, env_tensor.matvec(env_tensor.matvec(A)))
    # E_rms = math.sqrt(E_squared)
    # logging.debug("INFO: sqrt(<E^2>): {}".format(E_rms))
    return E, V


def compress_svd(U, S, V, m):
    """
    Parameters
    ----------
    U : (a, k) ndarray
    S : (k,) ndarray
    V : (k, b) ndarray
    m : int

    Returns
    -------
    U : (a, m) ndarray
    S : (m,) ndarray
    V : (m, b) ndarray
    compress_error : float
    """
    dim = min(len(S), m)
    compress_error = np.sum(S[dim:])
    S = S[:dim]
    U = U[:, :dim]
    V = V[:dim, :]
    return U, S, V, compress_error


def dmrg1(mpo, initial_mps, trunc=10, sweeps=8):
    r"""One-site DMRG method.

    Parameters
    ----------
    mpo : [(_, _, _, _) ndarray]
        A list of MPO matrixes, the Hamitonian of the system
    initial_mps : [(_, _, _) ndarray]
        a list of *right-canonical* MPS matrixes

    Returns
    -------
    E : float
        the energy of groundstate.
    mps : [(_, _, _) ndarray]
        a list of MPS matrixes, which is the wave function in MPS form.
    """
    _debug = logging.root.isEnabledFor(logging.DEBUG)

    def _move_right(mat, mat_next, _trunc):
        shape = np.shape(mat)
        # SVD on mat[si, j]
        mat = np.reshape(mat, (shape[0] * shape[1], shape[2]))
        U, S, V = np.linalg.svd(mat, full_matrices=0)

        # truncated to compress.
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
        mat = np.reshape(U, (shape[0], shape[1], -1))  # U[si, m]
        SV = np.matmul(np.diag(S), V)
        mat_next = np.einsum('mj,tjk->tmk', SV, mat_next)

        return mat, mat_next

    def _move_left(mat, mat_prev, _trunc):
        shape = np.shape(mat)
        mat = np.reshape(
            np.transpose(mat, (1, 0, 2)),  # mat[i, s, j]
            (shape[1], shape[0] * shape[2]))  # SVD on mat[i, sj]
        U, S, V = np.linalg.svd(mat, full_matrices=0)

        # truncated to compress.
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
        mat = np.reshape(V, (-1, shape[0], shape[2]))  # V[m, sj]
        mat = np.transpose(mat, (1, 0, 2))  # mat[s, m, j]
        US = np.matmul(U, np.diag(S))
        mat_prev = np.einsum('rhi,im->rhm', mat_prev, US)

        return mat, mat_prev

    def _contract_Rs(mpo, mps):
        R_list = [np.array([[[1.0]]])]
        for i in range(len(mpo) - 1, 0, -1):
            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))

        return R_list

    mps = initial_mps
    L_list = [np.array([[[1.0]]])]
    R_list = _contract_Rs(mpo, mps)

    for sweep in range(sweeps // 2):
        for i in range(len(mpo) - 1):  # right sweep
            L = L_list[-1]
            R = R_list.pop()
            H = EnvTensor(L, mpo[i], R)
            E, mps[i] = opt_one_site(H, mps[i])  # diag
            mps[i], mps[i + 1] = _move_right(mps[i], mps[i + 1], trunc)  # SVD
            L_list.append(contract_from_left(L_list[-1], mps[i], mpo[i], mps[i]))

            if _debug:
                logging.debug("Sweep {}, Sites {}, E1 {:16.12f}, E2 {:16.12f}".format(
                    sweep * 2, i, E, mat_element(mps, mpo, mps)))

        for i in range(len(mps) - 1, 0, -1):  # left sweep
            R = R_list[-1]
            L = L_list.pop()
            H = EnvTensor(L, mpo[i], R)
            E, mps[i] = opt_one_site(H, mps[i])  # diag
            mps[i], mps[i - 1] = _move_left(mps[i], mps[i - 1], trunc)  # SVD
            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))

            if _debug:
                logging.debug("Sweep {}, Sites {}, E1 {:16.12f}, E2 {:16.12f}".format(
                    sweep * 2 + 1, i, E, mat_element(mps, mpo, mps)))

    return mps


def coarse_grain_mps(A, B):
    """Coarse-graining of two-site MPS into one site.::

            |st         |s    |t
        -i- C -k- = -i- A -j- B -k-

    Parameters
    ----------
    A : (s, i, j) ndarray
        A MPS matrix.
    B : (t, j, k) ndarray
        A MPS matrix.

    Returns
    -------
    C : (st, i, k) ndarray
        A MPS matrix.
    """
    shape_C = [A.shape[0] * B.shape[0], A.shape[1], B.shape[2]]
    C = np.einsum("sij,tjk->stik", A, B)
    return np.reshape(C, shape_C)


def coarse_grain_mpo(W, X):
    """Coarse-graining of two site MPO into one site.::

            |su         |s    |u
        -a- R -c- = -a- W -b- X -c-
            |tv         |t    |v

    Parameters
    ----------
    W : (a, b, s, t) ndarray
        A MPS matrix.
    X : (b, c, u, v) ndarray
        A MPS matrix.

    Returns
    -------
    R : (a, c, su, tv) ndarray
        A MPS matrix.
     """

    R = np.einsum("abst,bcuv->acsutv", W, X)
    sh = [
        W.shape[0],  # a
        X.shape[1],  # c
        W.shape[2] * X.shape[2],  # su
        W.shape[3] * X.shape[3]
    ]  # tv
    return np.reshape(R, sh)


def fine_grain_mps(C, dims, direction, _trunc=False):
    """Fine-graining of one-site MPS into three site by SVD.::

            |st         |s    |t
        -i- C -k- = -i- A -m- B -k-

    Parameters
    ----------
    C : (st, i, k) ndarray
        A MPS matrix.
    dims : (int, int)
        [s, t].
    direction : {'>', '<'}
        '>': move to right; '<'  move to left
    _trunc : int, optional
        Set m in compress_svd to _trunc. Not compressed if not _trunc.

    Returns
    -------
    A : (s, i, m) ndarray
        A MPS matrix.
    B : (t, m, k) ndarray
        A MPS matrix.

    Notes
    -----
    If direction == '>', A is (left-)canonical;
    if direction == '<', B is (right-)canonical.
    """
    sh = dims + [C.shape[1], C.shape[2]]  # [s, t, i, k]
    mat = np.reshape(C, sh)
    mat = np.transpose(mat, (0, 2, 1, 3))  # mat[s, i, t, k]
    mat = np.reshape(mat, (sh[0] * sh[2], sh[1] * sh[3]))  # mat[si, tk]
    U, S, V = np.linalg.svd(mat, full_matrices=0)
    if _trunc:
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
    if direction == '>':
        A = U
        B = np.matmul(np.diag(S), V)  # [m, tk]
    elif direction == '<':
        A = np.matmul(U, np.diag(S))
        B = V
    A = np.reshape(A, (sh[0], sh[2], -1))  # [s, i, m]
    B = np.reshape(B, (-1, sh[1], sh[3]))  # [m, t, k]
    B = np.transpose(B, (1, 0, 2))  # [t, m, k]

    return A, B


def dmrg2(mpo, initial_mps, trunc=10, sweeps=8):
    r"""Two-site DMRG method.

    Parameters
    ----------
    mpo : [(_, _, _, _) ndarray]
        A list of MPO matrixes, the Hamitonian of the system
    initial_mps : [(_, _, _) ndarray]
        a list of *right-canonical* MPS matrixes

    Returns
    -------
    E : float
        the energy of groundstate.
    mps : [(_, _, _) ndarray]
        a list of MPS matrixes, which is the wave function in MPS form.
    """
    _debug = logging.root.isEnabledFor(logging.DEBUG)

    def _contract_Rs(mpo, mps):
        R_list = [np.array([[[1.0]]])]
        for i in range(len(mpo) - 1, 1, -1):  # contract from end to site 2
            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))
        return R_list

    def _contract_cg_mpo_list(mpo):
        cg_mpo_list = []
        for i in range(len(mpo) - 1):
            cg_mpo_list.append(coarse_grain_mpo(mpo[i], mpo[i + 1]))
        return cg_mpo_list

    mps = initial_mps
    L_list = [np.array([[[1.0]]])]
    R_list = _contract_Rs(mpo, mps)
    # cg_mpo_list = _contract_cg_mpo_list(mpo)

    for sweep in range(sweeps // 2):
        for i in range(len(mpo) - 2):  # right sweep
            L = L_list[-1]
            R = R_list.pop()
            fg_dims = [mps[i].shape[0], mps[i + 1].shape[0]]
            cg_mps = coarse_grain_mps(mps[i], mps[i + 1])
            cg_mpo = coarse_grain_mpo(mpo[i], mpo[i + 1])
            H = EnvTensor(L, cg_mpo, R)
            E, cg_mps = opt_one_site(H, cg_mps)
            mps[i], mps[i + 1] = fine_grain_mps(cg_mps, fg_dims, '>', trunc)

            L_list.append(contract_from_left(L_list[-1], mps[i], mpo[i], mps[i]))

            if _debug:
                logging.debug("Sweep {}, Sites {}, {}, E1 {:16.12f}, E2 {:16.12f}".format(
                    sweep * 2, i, i + 1, E, mat_element(mps, mpo, mps)))

        for i in range(len(mps) - 1, 1, -1):  # left sweep
            R = R_list[-1]
            L = L_list.pop()
            fg_dims = [mps[i - 1].shape[0], mps[i].shape[0]]
            cg_mps = coarse_grain_mps(mps[i - 1], mps[i])
            cg_mpo = coarse_grain_mpo(mpo[i - 1], mpo[i])
            H = EnvTensor(L, cg_mpo, R)
            E, cg_mps = opt_one_site(H, cg_mps)
            mps[i - 1], mps[i] = fine_grain_mps(cg_mps, fg_dims, '<', trunc)

            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))

            if _debug:
                logging.debug("Sweep {}, Sites {}, {}, E1 {:16.12f}, E2 {:16.12f}".format(
                    sweep * 2 + 1, i - 1, i, E, mat_element(mps, mpo, mps)))

    return mps


def mat_element(mps1, mpo, mps2):
    r"""Calculate the matrix element :math:`\langle 1 | O | 2 \rangle` by
    contract_from_left, where :math:`| 1 \rangle`, :math:`| 2 \rangle` are MPS,
    :math:`O` is MPO.

    Parameters
    mps1 : [(_, _, _) ndarray]
        a list of MPS matrixes
    mpo : [(_, _, _, _) ndarray]
        a list of MPO matrixes or None
    mps2 : [(_, _, _) ndarray]
        a list of MPS matrixes

    Returns
    -------
        Matrix element :math:`\langle 1 | O | 2 \rangle` or
        :math:`\langle 1 | 2 \rangle` if mpo is None
    """
    if mpo is not None:
        L = np.array([[[1.0]]])
        for i in range(0, len(mpo)):
            L = contract_from_left(L, mps1[i], mpo[i], mps2[i])
        return L[0, 0, 0]
    else:
        L = np.array([[1.0]])
        assert len(mps1) == len(mps2)
        for i in range(0, len(mps1)):
            L = contract_from_left(L, mps1[i], None, mps2[i])
        return L[0, 0]
