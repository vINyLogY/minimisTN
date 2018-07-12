#!/usr/bin/env python2
# coding: utf-8

r"""# A Simple MPS/MPO Program

by Xinxian Chen
(chen-xx15@mails.tsinghua.edu.cn)


## Definition

- MPS matrix: a 3-index tensor A[s, i, j]

            |s
        -i- A -j-

    [s]:    local Hilbert space,
    [i,j]:  virtual indices.

- MPO matrix: a 4-index tensor W[a, b, s, t]

            |s
        -a- W -b-
            |t

    [s, t]: local Hilbert space,
    [a, b]: virtual indices.


## Reference

1. arXiv:1008.3477v2
2. arXiv:1603.03039v4
"""

import math

import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg

import dmrg

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
    up = np.zeros((2,1,1))
    up[0,0,0] = 1
    down = np.zeros((2,1,1))
    down[1,0,0] = 1
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
    Sz = np.array([[0.5,  0. ],
                   [0. , -0.5]])
    Sp = np.array([[0., 0.],
                   [1., 0.]])
    Sm = np.array([[0., 1.],
                   [0., 0.]])
    # left-hand edge: 1*5
    Wfirst = np.array([[-h*Sz,  (J/2.)*Sm, (J/2.)*Sp, (Jz/2.)*Sz, I]])
    # mid: 5*5
    W = np.array([[    I,          Z,         Z,          Z, Z],
                  [   Sp,          Z,         Z,          Z, Z], 
                  [   Sm,          Z,         Z,          Z, Z],
                  [   Sz,          Z,         Z,          Z, Z],
                  [-h*Sz,  (J/2.)*Sm, (J/2.)*Sp, (Jz/2.)*Sz, I]])
    # right-hand edge: 5*1
    Wlast = np.array([[    I],
                      [   Sp],
                      [   Sm],
                      [   Sz],
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
    - W: a MPO matrix W[a, b, s, t]
    - B: a MPS matrix B[t, k, l]

    ## Returns
    - L': a 3-index tensor L'[b, j, l]
    """
    Temp = np.einsum('sij,aik->sajk', A, L)
    Temp = np.einsum('sajk,abst->tbjk', Temp, W)
    return np.einsum('tbjk,tkl->bjl', Temp, B)

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
    - W: a MPO matrix W[a, b, s, t]
    - B: a MPS matrix B[t, k, l]

    ## Returns
    - R': a 3-index tensor R'[a, i, k]
    """

    temp = np.einsum('sij,bjl->sbil', A, R)
    temp = np.einsum('sbil,abst->tail', temp, W)
    return np.einsum('tail,tkl->aik', temp, B)

class EnvTensor(sparse.linalg.LinearOperator):
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

def opt_one_site(H, A):
    r"""
    Solve eigenvalue problem at one site A.

    ## Args
    - H: a linear operator, which is the environment tensor of A.
        A[t, j, l] --> (HA)[s, i, k]
    - A: a MPS matrix A[t, j, l]
    """

    A = np.reshape(A, H.size)
    E, V = scipy.sparse.linalg.eigsh(H, 1, v0=A, which='SA')
    return (E[0], np.reshape(V[:, 0], H.io_shape))


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
        mat = np.reshape(mat, (shape[0]*shape[1], shape[2]))    # SVD on mat[si, j]
        U, S, V = np.linalg.svd(mat, full_matrices=0)

        # truncated to compress.
        _trunc = min(len(S), _trunc)
        mat = np.reshape(U[:, :_trunc], (shape[0], shape[1], _trunc))    # U[si, m]
        S = S[:_trunc]    # m dims
        V = V[:_trunc, :]    # V[m, j]
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
        _trunc = min(len(S), _trunc)
        mat = np.reshape(V[:_trunc, :], (_trunc, shape[0], shape[2]))  # V[m, s, j]
        mat = np.transpose(mat, (1, 0, 2))    # mat[s, m, j]  
        S = S[:_trunc]    # m dims
        U = U[:, :_trunc]    # U[i, m]
        US = np.matmul(U, np.diag(S))
        mat_prev = np.einsum('rhi,im->rhm', mat_prev, US)

        return mat, mat_prev

    def _contract_Rs(mpo, mps):
        R_list = [np.array([[[1.0]]])]
        for i in range(len(mpo)-1, 0, -1):
            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))
        
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
            mps[i], mps[i+1] = _move_right(mps[i], mps[i+1], trunc)   #SVD
            L_list.append(contract_from_left(L_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {} Energy {:16.12f}".format(sweep*2, i , E))

        for i in range(len(mps)-1, 0, -1):    # left sweep
            R = R_list[-1]
            L = L_list.pop()
            H = EnvTensor(L, mpo[i], R)
            E, mps[i] = opt_one_site(H, mps[i])   # diag
            mps[i], mps[i-1] = _move_left(mps[i], mps[i-1], trunc)   #SVD
            R_list.append(contract_from_right(R_list[-1], mps[i], mpo[i], mps[i]))

            print("Sweep {}, Sites {} Energy {:16.12f}".format(sweep*2+1, i , E))

    return mps


def dmrg2(*arg, **kwarg):
    r"""Two-site DMRG method.
    """
    
    # TODO

    return


def mat_element(mps1, mpo, mps2):
    r"""Calculate the matrix element <1|O|2>,
    where |1>, |2> are MPS, O is MPO.
    
    ## Args
    - mps1: a list of MPS matrixes
    - mpo: a list of MPO matrixes 
    - mps2: a list of MPS matrixes

    ## Returns
    - Matrix element <mps1|mpo|mps2>
    """

    L = np.array([[[1.0]]])
    for i in range(0,len(mpo)):
        L = contract_from_left(L, mps1[i], mpo[i], mps2[i])
    return L[0, 0, 0]


def main():
    r"""
    Test program for Heisenberg Model.
    """

    N = 10
    initial_mps = fm_state(N, anti=1)
    mpo = heisenberg(N)

    mps = dmrg.two_site_dmrg(initial_mps, mpo, 10, 4)
    for i in range(len(mps)):
        print("MPS[{}] shape: {}".format(i, np.shape(mps[i])))

    # 8 sweeps with m=10 states
    mps = dmrg1(mpo, mps, 10, 6)
    # energy and energy squared
    energy = mat_element(mps, mpo, mps)

    mps = dmrg.two_site_dmrg(initial_mps, mpo, 10, 10)
    energy_ref = dmrg.Expectation(mps, mpo, mps)

    print('Final energy expectation value {}'.format(energy))
    print("Reference: {}".format(energy_ref))



    return

if __name__=="__main__":
    main()
