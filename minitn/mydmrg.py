#!/usr/bin/env python2
# coding: utf-8

r"""# A Simple MPS/MPO Program

## Definition

- MPS matrix: a 3-index tensor A[s, i, j]

            |s
        -i- A -j-

    [s]: local Hilbert space,
    [i,j]: virtual indices.

- MPO matrix: a 4-index tensor W[a, b, s, t]

            |s
        -a- W -b-
            |t

    [s, t]: local Hilbert space,
    [a, b]: virtual indices.

## Reference

1. arXiv:1008.3477v2
2. arXiv:1603.03039v4
3. https://people.smp.uq.edu.au/IanMcCulloch/mptoolkit/index.php
"""

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


def fm_state(N, anti=0):
    r"""
    Get a 1-D n-site antiferromagnetic state |+-+- ... +->
    or ferromagnetic state |++ ... +>

    ## Args
    - N: (even) number of sites.

    ## Returns
    - A list of MPS matrixes.
    """
    # local bond dimension, 0=up, 1=down
    up = np.zeros((2, 1, 1))
    up[0, 0, 0] = 1
    down = np.zeros((2, 1, 1))
    down[1, 0, 0] = 1
    if anti == 1:
        mps = [up, down] * int(N/2)
    else:
        mps = [up] * N
    return mps


def heisenberg(N, J=1.0, Jz=1.0, h=0):
    r"""Generate a MPO for Heisenberg Model

    $$
    H = \sum^{N-2}_{i=0} \frac{J}{2} (S^+_i S^-_{i+1} + S^-_i S^+_{i+1})
        + J_z S^z_i S^z_{i+1}
        - \sum^{N-1}_{i=0} h S^z_i
    $$

    For 1-D antiferromagnetic, $J = J_z = 1$.

    ## Args
    - N: number of sites.
    - J: coupling constant.
    - Jz: coupling constant in z-direction.
    - h: external magnetic field.

    ## Returns
    - A list of MPO matrixes.
    """

    # Local operators
    I = np.eye(2)
    Z = np.zeros((2, 2))
    Sz = np.array([[0.5,  0.0],
                   [0.0, -0.5]])
    Sp = np.array([[0., 0.],
                   [1., 0.]])
    Sm = np.array([[0., 1.],
                   [0., 0.]])
    # left-hand edge: 1*5
    Wfirst = np.array([[-h*Sz,  (J/2.)*Sm, (J/2.)*Sp, (Jz/2.)*Sz, I]])
    # mid: 5*5
    W = np.array([[I,           Z,         Z,          Z, Z],
                  [Sp,          Z,         Z,          Z, Z],
                  [Sm,          Z,         Z,          Z, Z],
                  [Sz,          Z,         Z,          Z, Z],
                  [-h*Sz,  (J/2.)*Sm, (J/2.)*Sp, (Jz/2.)*Sz, I]])
    # right-hand edge: 5*1
    Wlast = np.array([[I],
                      [Sp],
                      [Sm],
                      [Sz],
                      [-h*Sz]])
    mpo = [Wfirst] + ([W] * (N-2)) + [Wlast]

    return mpo


def contract_from_left(L, A, W, B):
    r"""Tensor contraction from the left hand side.

        + -j-     + -i- A -j-
        |         |     |s
        L'-b-  =  L -a- W -b-
        |         |     |t
        + -l-     + -k- B -l-

    ## Args:
    - L: a 3-index tensor L[a, i, k]
    - A: a MPS matrix A[s, i, j]
    - W: a MPO matrix W[a, b, s, t], or None
    - B: a MPS matrix B[t, k, l]

    ## Returns
    - L': a 3-index tensor L'[b, j, l],
        or L'[j, l] if W is None
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
    r"""Tensor contraction from the right hand side.

        -i- +    -i- A -j- +
            |        |s    |
        -a- R' = -a- W -b- R
            |        |t    |
        -k- +    -k- B -l- +

    ## Args:
    - R: a 3-index tensor R[b, j, l]
    - A: a MPS matrix A[s, i, j]
    - W: a MPO matrix W[a, b, s, t] or None
    - B: a MPS matrix B[t, k, l]

    ## Returns
    - R': a 3-index tensor R'[a, i, k]
        or R'[i, k] if W is None
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
    r"""An environment tensor to evaluate the Hamiltonian matrix-vector multiply

        i s k   + -i-   -k- +
        | | |   |     |s    |
        + H + = L -a- W -b- R
        | | |   |     |t    |
        j t l   + -j-   -l- +

    ## Args
    - L: a 3-index tensor L[a, i, j]
    - W: a MPO matrix W[a, b, s, t]
    - R: a 3-index tensor R[b, k, l]
    """

    def __init__(self, L, W, R):
        self.L = L
        self.W = W
        self.R = R
        self.io_shape = [W.shape[3], L.shape[2], R.shape[2]]
        self.dtype = np.dtype('d')
        self.size = self.io_shape[0] * self.io_shape[1] * self.io_shape[2]
        self.shape = [self.size] * 2

    def _matvec(self, A):
        r"""Default matrix-vector multiplication.

        ## Args
        - A: vector, the length of which should be 'size'

        ## Returns
        - A vector, the length of which should be 'size'
        """

        temp = np.einsum('aij,sik->ajsk', self.L, np.reshape(A, self.io_shape))
        temp = np.einsum('ajsk,abst->bjtk', temp, self.W)
        temp = np.einsum('bjtk,bkl->tjl', temp, self.R)
        return np.reshape(temp, self.size)


def opt_one_site(env_tensor, A):
    r"""
    Solve eigenvalue problem at one site A.

    ## Args
    - env_tensor: a linear operator, which is the environment tensor of A.
        A[t, j, l] --> (env_tensor(A))[s, i, k]
    - A: a MPS matrix A[t, j, l]

    ## Return
    - E: Rayleigh quotient
    - V: optimized one-site MPS matrix
    """

    A = np.reshape(A, env_tensor.size)
    E, V = eigsh(env_tensor, 1, v0=A, which='SA')
    E, V = E[0], np.reshape(V[:, 0], env_tensor.io_shape)
    # A = np.reshape(V, env_tensor.size)
    # E_squared = np.dot(A, env_tensor.matvec(env_tensor.matvec(A)))
    # E_rms = math.sqrt(E_squared)
    # print("INFO: sqrt(<E^2>): {}".format(E_rms))

    return E, V


def compress_svd(U, S, V, m):
    dim = min(len(S), m)
    compress_error = np.sum(S[dim:])
    S = S[:dim]
    U = U[:, :dim]
    V = V[:dim, :]
    return U, S, V, compress_error


def dmrg1(mpo, initial_mps, trunc=10, sweeps=8):
    r"""One-site DMRG method

    ## Args
    - mpo: a list of MPO matrixes, the Hamitonian of the system
    - initial_mps: a list of *right-canonical* MPS matrixes

    ## Returns
    - E: the energy of groundstate.
    - mps: a list of MPS matrixes, which is the wave function in MPS form.
    """

    def _move_right(mat, mat_next, _trunc):
        r"""Move to the right site.

        ## Args:
        - mat: a MPS matrix A[s, i, j]
        - mat_next: a MPS matrix B[t, j, k]
        """

        shape = np.shape(mat)
        # SVD on mat[si, j]
        mat = np.reshape(mat, (shape[0]*shape[1], shape[2]))
        U, S, V = np.linalg.svd(mat, full_matrices=0)

        # truncated to compress.
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
        mat = np.reshape(U, (shape[0], shape[1], -1))    # U[si, m]
        SV = np.matmul(np.diag(S), V)
        mat_next = np.einsum('mj,tjk->tmk', SV, mat_next)

        return mat, mat_next

    def _move_left(mat, mat_prev, _trunc):
        r"""Move to the left site.

        ## Args:
        - mat: a MPS matrix A[s, i, j]
        - mat_prev: a MPS matrix B[r, h, i]
        """

        shape = np.shape(mat)
        mat = np.reshape(np.transpose(mat, (1, 0, 2)),    # mat[i, s, j]
                         (shape[1], shape[0]*shape[2]))    # SVD on mat[i, sj]
        U, S, V = np.linalg.svd(mat, full_matrices=0)

        # truncated to compress.
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
        mat = np.reshape(V, (-1, shape[0], shape[2]))    # V[m, sj]
        mat = np.transpose(mat, (1, 0, 2))    # mat[s, m, j]
        US = np.matmul(U, np.diag(S))
        mat_prev = np.einsum('rhi,im->rhm', mat_prev, US)

        return mat, mat_prev

    def _contract_Rs(mpo, mps):
        R_list = [np.array([[[1.0]]])]
        for i in range(len(mpo)-1, 0, -1):
            R_list.append(contract_from_right(
                R_list[-1], mps[i], mpo[i], mps[i]))

        return R_list

    mps = initial_mps
    L_list = [np.array([[[1.0]]])]
    R_list = _contract_Rs(mpo, mps)

    for sweep in range(sweeps/2):
        for i in range(len(mpo)-1):    # right sweep
            L = L_list[-1]
            R = R_list.pop()
            H = EnvTensor(L, mpo[i], R)
            E, mps[i] = opt_one_site(H, mps[i])   # diag
            mps[i], mps[i+1] = _move_right(mps[i], mps[i+1], trunc)  # SVD
            L_list.append(contract_from_left(
                L_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {}, E1 {:16.12f}, E2 {:16.12f}".format(
                sweep*2, i, E, mat_element(mps, mpo, mps)))

        for i in range(len(mps)-1, 0, -1):    # left sweep
            R = R_list[-1]
            L = L_list.pop()
            H = EnvTensor(L, mpo[i], R)
            E, mps[i] = opt_one_site(H, mps[i])   # diag
            mps[i], mps[i-1] = _move_left(mps[i], mps[i-1], trunc)  # SVD
            R_list.append(contract_from_right(
                R_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {}, E1 {:16.12f}, E2 {:16.12f}".format(
                sweep*2 + 1, i, E, mat_element(mps, mpo, mps)))

    return mps


def coarse_grain_mps(A, B):
    """Coarse-graining of two-site MPS into one site

            |st         |s    |t
        -i- C -k- = -i- A -j- B -k-

    ## Args
    - A: a MPS matrix A[s, i, j]
    - B: a MPS matrix B[t, j, k]

    ## Return
    - C: a MPS matrix C[st, i, k]
    """
    shape_C = [A.shape[0] * B.shape[0], A.shape[1], B.shape[2]]
    C = np.einsum("sij,tjk->stik", A, B)
    return np.reshape(C, shape_C)


def coarse_grain_mpo(W, X):
    """Coarse-graining of two site MPO into one site
            |su         |s    |u
        -a- R -c- = -a- W -b- X -c-
            |tv         |t    |v
     """

    R = np.einsum("abst,bcuv->acsutv", W, X)
    sh = [W.shape[0],    # a
          X.shape[1],    # c
          W.shape[2] * X.shape[2],    # su
          W.shape[3] * X.shape[3]]    # tv
    return np.reshape(R, sh)


def fine_grain_mps(C, dims, direction, _trunc=False):
    """Fine-graining of one-site MPS into three site by SVD

            |st         |s    |t
        -i- C -k- = -i- A -m- B -k-

    ## Args
    - C: a MPS matrix C[st, i, k]
    - dims: [s, t]
    - direction: char
        '>': move to right; '<'  move to left
    - _trunc: if _truc != None, m = _trunc

    ## Return
    - A: a MPS matrix A[s, i, m]
    - B: a MPS matrix B[t, m, k]

    If direction == '>', A is (left-)canonical;
    if direction == '<', B is (right-)canonical
    """

    sh = dims + [C.shape[1], C.shape[2]]    # [s, t, i, k]
    mat = np.reshape(C, sh)
    mat = np.transpose(mat, (0, 2, 1, 3))    # mat[s, i, t, k]
    mat = np.reshape(mat, (sh[0] * sh[2], sh[1] * sh[3]))        # mat[si, tk]
    U, S, V = np.linalg.svd(mat, full_matrices=0)
    if _trunc:
        U, S, V, compress_error = compress_svd(U, S, V, _trunc)
    if direction == '>':
        A = U
        B = np.matmul(np.diag(S), V)    # [m, tk]
    elif direction == '<':
        A = np.matmul(U, np.diag(S))
        B = V
    A = np.reshape(A, (sh[0], sh[2], -1))    # [s, i, m]
    B = np.reshape(B, (-1, sh[1], sh[3]))    # [m, t, k]
    B = np.transpose(B, (1, 0, 2))    # [t, m, k]

    return A, B


def dmrg2(mpo, initial_mps, trunc=10, sweeps=8):
    r"""Two-site DMRG method.

    ## Args:
    - mpo: a list of MPO matrix
    - initial_mps: a list of MPS matrix

    ## Return:
    - mpo: a list of MPO matrix
    """

    def _contract_Rs(mpo, mps):
        R_list = [np.array([[[1.0]]])]
        for i in range(len(mpo)-1, 1, -1):    # contract from end to site 2
            R_list.append(contract_from_right(
                R_list[-1], mps[i], mpo[i], mps[i]))
        return R_list

    def _contract_cg_mpo_list(mpo):
        cg_mpo_list = []
        for i in range(len(mpo)-1):
            cg_mpo_list.append(coarse_grain_mpo(mpo[i], mpo[i+1]))
        return cg_mpo_list

    mps = initial_mps
    L_list = [np.array([[[1.0]]])]
    R_list = _contract_Rs(mpo, mps)
    # cg_mpo_list = _contract_cg_mpo_list(mpo)

    for sweep in range(sweeps/2):
        for i in range(len(mpo)-2):    # right sweep
            L = L_list[-1]
            R = R_list.pop()
            fg_dims = [mps[i].shape[0], mps[i+1].shape[0]]
            cg_mps = coarse_grain_mps(mps[i], mps[i+1])
            cg_mpo = coarse_grain_mpo(mpo[i], mpo[i+1])
            H = EnvTensor(L, cg_mpo, R)
            E, cg_mps = opt_one_site(H, cg_mps)
            mps[i], mps[i+1] = fine_grain_mps(cg_mps, fg_dims, '>', trunc)

            L_list.append(contract_from_left(
                L_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {}, {}, E1 {:16.12f}, E2 {:16.12f}".format(
                sweep*2, i, i+1, E, mat_element(mps, mpo, mps)))

        for i in range(len(mps)-1, 1, -1):    # left sweep
            R = R_list[-1]
            L = L_list.pop()
            fg_dims = [mps[i-1].shape[0], mps[i].shape[0]]
            cg_mps = coarse_grain_mps(mps[i-1], mps[i])
            cg_mpo = coarse_grain_mpo(mpo[i-1], mpo[i])
            H = EnvTensor(L, cg_mpo, R)
            E, cg_mps = opt_one_site(H, cg_mps)
            mps[i-1], mps[i] = fine_grain_mps(cg_mps, fg_dims, '<', trunc)

            R_list.append(contract_from_right(
                R_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {}, {}, E1 {:16.12f}, E2 {:16.12f}".format(
                sweep*2+1, i-1, i, E, mat_element(mps, mpo, mps)))

    return mps


def mat_element(mps1, mpo, mps2):
    r"""Calculate the matrix element <1|O|2> by contract from left,
    where |1>, |2> are MPS, O is MPO.

    ## Args
    - mps1: a list of MPS matrixes
    - mpo: a list of MPO matrixes or None
    - mps2: a list of MPS matrixes

    ## Returns
    - Matrix element <mps1|mpo|mps2>
        or <mps1|mps2> if mpo is None
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


def main():
    r"""
    Test program for Heisenberg Model.
    """

    N = 20
    mpo = heisenberg(N)

    mps = dmrg2(mpo, fm_state(N, anti=1), 10, 2)
    mps = dmrg1(mpo, mps, 10, 8)
    energy1 = mat_element(mps, mpo, mps)

    mps = dmrg2(mpo, fm_state(N, anti=1), 10, 10)
    energy2 = mat_element(mps, mpo, mps)

    print('DMRG1 energy: {}'.format(energy1))
    print('DMRG2 energy: {}'.format(energy2))

    return


if __name__ == '__main__':
    main()
