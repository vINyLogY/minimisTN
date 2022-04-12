#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from asyncio.log import logger

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


def test_heom(fname=None,
              dof=3,
              max_tier=10,
              rank_heom=10,
              k_max=6,
              decomp_method=None,
              scale=1.0,
              coupling=2500,
              temperature=100_000,
              ps_method='split',
              ode_method='RK45'):
    # System:
    e = Quantity(5000, 'cm-1').value_in_au
    relaxed = True
    v = Quantity(500, 'cm-1').value_in_au if relaxed else 0.0

    rank_heom = rank_heom if decomp_method is not None else None

    beta = Quantity(1 /
                    temperature, 'K-1').value_in_au if temperature else None
    sd_method = Drude.pade
    w = Quantity(50, 'cm-1').value_in_au
    drude = Drude(gamma=w,
                  lambda_=Quantity(500, 'cm-1').value_in_au,
                  beta=beta,
                  k_max=k_max,
                  decompmethod=sd_method)

    model = SpinBoson(
        sys_ham=np.array([[0.0, v], [v, e]], dtype=DTYPE),
        sys_op=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
        ph_parameters=[],
        ph_dims=[],
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
        f'boson-drude_{"relaxed" if relaxed else "pure"}_'
        f'{decomp_method}_{temperature}K_dof{dof}_bcf{k_max}_cp{coupling}_'
        f't{max_tier}_r{rank_heom}_{ps_method}_{ode_method}')
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

    leaves = root.leaves()
    h_list = model.heom_h_list(leaves[0],
                               leaves[1],
                               bath_indices=leaves[2:],
                               beta=beta,
                               scale=scale)

    solver = MultiLayer(root, h_list)
    solver.ode_method = ode_method
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    if ps_method is not None:
        solver.ps_method = ps_method
    solver.svd_err = 1.0e-10  #only for unite propagation

    # DVR

    length = 100
    bath_basis = SineDVR(-length, length, 1000)
    bath_basis.set_v_func(lambda x: 0.5 * x**2)
    eig_v, u_mat = np.linalg.eigh(bath_basis.h_mat())
    eig_v, u_mat = eig_v[:max_tier], np.transpose(u_mat[:, :max_tier])
    grids = bath_basis.grid_points

    # Define the obersevable of interest

    cpu_t0 = cpu_time()
    logger1 = Logger(filename=fname, level='info').logger
    logger2 = Logger(filename='DEBUG_' + fname, level='info').logger
    logger2.info("# time    CPU_time")
    for n, (time, r) in enumerate(
            solver.propagator(steps=count,
                              ode_inter=dt_unit,
                              split=True if ps_method is not None else False)):
        rho = np.reshape(r.array, (2, 2, max_tier, -1))
        rho = Tensor.partial_product(rho, 2, u_mat)

        plt.plot(grids, np.abs(np.real(rho[0, 0, :, 0])), 'k.', label='Pop.')
        plt.plot(grids, np.abs(rho[0, 1, :, 0]), 'rx', label='Coh.')
        plt.xlim(-10, 10)
        plt.ylim(-0.5, 0.5)
        plt.savefig(f'{n:08d}.png')
        plt.close()

        rv = np.reshape(r.array, (4, -1))[:, 0]
        logger1.info("{}    {} {} {} {}".format(
            time,
            rv[0],
            rv[1],
            rv[2],
            rv[3],
        ))
        print(f'Coh: {np.abs(rv[1])}')
    return


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'drude_std'))

    for coupling in [1000]:
        test_heom(
            fname=f'heom.dat',
            dof=0,
            max_tier=50,
            decomp_method=None,
            temperature=300,
            k_max=1,
            coupling=coupling,
            ode_method='RK45',
            ps_method=None,
            scale=1.0,
        )
