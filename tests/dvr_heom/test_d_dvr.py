#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from time import time as cpu_time

from matplotlib import pyplot as plt

from minitn.heom.corr import Drude
from minitn.heom.network import simple_heom, tensor_train_template
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.units import Quantity
from minitn.tensor import Tensor
from minitn.bases.dvr import SineDVR
from minitn.models.sbm import SpinBoson


def test_heom(
    fname=None,
    dof=3,
    max_tier=10,
    rank_heom=10,
    k_max=6,
    decomp_method=None,
    scale=1.0,
    coupling=1500,
    length=50,
    temperature=100_000,
    ps_method='split',
    ode_method='RK45',
):
    # System:
    e = Quantity(5000, 'cm-1').value_in_au
    relaxed = False
    v = Quantity(2500, 'cm-1').value_in_au if relaxed else 0.0

    rank_heom = rank_heom if decomp_method is not None else None

    beta = Quantity(1 /
                    temperature, 'K-1').value_in_au if temperature else None
    w = Quantity(2500, 'cm-1').value_in_au
    g = Quantity(coupling, 'cm-1').value_in_au
    ph_parameters = [(w, g)][:dof]

    sd_method = Drude.pade
    drude = Drude(gamma=Quantity(50, 'cm-1').value_in_au,
                  lambda_=Quantity(500, 'cm-1').value_in_au,
                  beta=beta,
                  k_max=k_max,
                  decompmethod=sd_method)

    model = SpinBoson(
        sys_ham=np.array([[0.0, v], [v, e]], dtype=DTYPE),
        sys_op=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
        ph_parameters=ph_parameters,
        ph_dims=([max_tier] * dof),
        bath_corr=drude,
        bath_dims=[max_tier] * k_max,
    )

    # init state
    A, B = 1.0, 1.0
    wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
    rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

    # Propagation
    dt_unit = Quantity(.1, 'fs').value_in_au
    count = 1000

    prefix = (
        f'drude_DVR_{"relaxed" if relaxed else "pure"}_'
        f'{decomp_method}_{temperature}K_dof{dof}_bcf{k_max}_cp{coupling}_'
        f't{max_tier}_r{rank_heom}_{ps_method}_{ode_method}_f{scale}')
    print(prefix)

    fname = prefix + '_' + fname
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    if decomp_method is None:
        root = simple_heom(rho_0, n_dims)
    elif decomp_method == 'TT':
        root = tensor_train_template(rho_0, n_dims, rank=rank_heom)
    else:
        raise NotImplementedError

    # DVR
    bath_basis = SineDVR(-length, length, max_tier)
    bath_basis.set_v_func(lambda x: 0.5 * x**2)

    def _eig_mat():
        w, u = np.linalg.eigh(bath_basis.h_mat())
        # correct the direction according to a+ and a-
        a = np.transpose(
            (bath_basis.q_mat() - bath_basis.dq_mat()) / np.sqrt(2.0))
        diag = np.diagonal(u.T @ a @ u, offset=1)
        counter = 0
        for _n, _d in enumerate(diag):
            _n_ = _n + 1
            if _d < 0.0:
                counter += 1
            u[:, _n_] = (-1)**counter * u[:, _n_]

        # print(f"Eigen values: {w[:10]}")
        return u

    bath_basis.eig_mat = _eig_mat
    grids = bath_basis.grid_points

    # Prepare EOM
    leaves = root.leaves()
    h_list = model.heom_h_list(leaves[0],
                               leaves[1],
                               bath_indices=leaves[2:],
                               beta=beta,
                               basis=[bath_basis, bath_basis],
                               scale=scale)

    # Prepare initial state
    tfmat = np.transpose(bath_basis.eig_mat())
    array = root.array
    for _i in [2]:
        array = Tensor.partial_product(array, _i, tfmat)

    # Prepare HEOM solver
    root.set_array(array)
    solver = MultiLayer(root, h_list)
    solver.ode_method = ode_method
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    if ps_method is not None:
        solver.ps_method = ps_method
    solver.svd_err = 1.0e-10  #only for unite propagation

    logger1 = Logger(filename=fname, level='info').logger
    #logger2.info("# time    CPU_time")
    for n, (time, r) in enumerate(
            solver.propagator(steps=count, ode_inter=dt_unit, split=False)):

        rho0 = np.reshape(r.array, (2, 2, max_tier, -1))[:, :, :, 0]
        plt.plot(grids, np.real(rho0[0, 0, :]), 'k.', label='Pop.')
        plt.plot(grids, np.abs(rho0[0, 1, :]), 'rx', label='Coh.')
        plt.legend()
        plt.xlim(-10, 10)
        plt.ylim(-.5, .5)
        plt.savefig(f'{n:08d}.png')
        plt.close()

        rho0 = Tensor.partial_product(rho0, 2, np.transpose(tfmat))
        rv = np.reshape(rho0, (4, -1))
        logger1.info(f"{time}    {rv[0, 0]} {rv[1, 0]} {rv[2, 0]} {rv[3, 0]}")
        print(f'Coh: {np.abs(rv[1, 0])}')
        # logger2.info("{} {}".format(
        #     time,
        #     cpu_time() - cpu_t0,
        # ))

    return


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'drude_dvr'))

    L = 100
    DL = 0.1
    N = int(L / DL)
    print(f"Max tier: {N}")

    test_heom(
        fname=f'heom.dat',
        dof=0,
        length=L,
        max_tier=N,
        coupling=1000,
        decomp_method=None,
        k_max=1,
        temperature=0,
        ode_method='RK45',
        ps_method=None,
        scale=1.0,
    )
