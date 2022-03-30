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


def test_heom(
    fname=None,
    dof=3,
    max_tier=10,
    rank_heom=10,
    k_max=6,
    decomp_method=None,
    scale=1.0,
    coupling=2500,
    temperature=300,
    ps_method='split',
    ode_method='RK45',
):
    # System:
    e = Quantity(5000, 'cm-1').value_in_au
    relaxed = True
    v = Quantity(500, 'cm-1').value_in_au if relaxed else 0.0

    rank_heom = rank_heom if decomp_method is not None else None

    beta = Quantity(1 /
                    temperature, 'K-1').value_in_au if temperature else None
    w = Quantity(2500, 'cm-1').value_in_au
    g = Quantity(coupling, 'cm-1').value_in_au
    ph_parameters = [(w, g)]

    sd_method = Drude.pade
    drude = Drude(gamma=Quantity(20, 'cm-1').value_in_au,
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
    dt_unit = Quantity(.01, 'fs').value_in_au
    callback_interval = 100
    count = 10000

    prefix = (
        f'boson_DVR_{"relaxed" if relaxed else "pure"}_'
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

    # DVR
    length = 5.0 / w
    plot_len = 15
    bath_basis = SineDVR(-length, length, max_tier)

    bath_basis.set_v_func(lambda x: 0.5 * (w * x)**2)
    eig_v, u_mat = np.linalg.eigh(bath_basis.h_mat())
    eig_v, u_mat = eig_v[:max_tier], np.transpose(u_mat[:, :max_tier])
    grids = bath_basis.grid_points

    leaves = root.leaves()
    array = root.array
    for _i in [2, 3]:
        array = Tensor.partial_product(array, _i, u_mat)
    root.set_array(array)
    h_list = model.heom_h_list(leaves[0],
                               leaves[1],
                               bath_indices=leaves[2:],
                               beta=beta,
                               basis=[bath_basis, bath_basis],
                               scale=scale)

    solver = MultiLayer(root, h_list)
    solver.ode_method = ode_method
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    if ps_method is not None:
        solver.ps_method = ps_method
    solver.svd_err = 1.0e-10  #only for unite propagation

    # Define the obersevable of interest
    levels = np.linspace(-0.035, 0.035, 350)
    cmap = "seismic"

    cpu_t0 = cpu_time()
    logger1 = Logger(filename=fname, level='info').logger
    logger2 = Logger(filename='DEBUG_' + fname, level='info').logger
    logger2.info("# time    CPU_time")
    for n, (time, r) in enumerate(
            solver.propagator(steps=count,
                              ode_inter=dt_unit,
                              split=True if ps_method is not None else False)):
        if n % callback_interval == 0:
            rho = r.array

            plt.plot(grids, np.real(rho[0, 0, :, 0]), label='Pop.')
            plt.plot(grids, np.abs(rho[0, 1, :, 0]), label='Coh.')
            plt.legend()
            plt.savefig(f'rho_{n}_{prefix}.png')
            plt.close()

            array = r.array
            for _i in [2, 3]:
                array = Tensor.partial_product(array, _i,
                                               np.transpose(u_mat[0:1, :]))
            rv = np.reshape(array, (4, ))
            logger1.info("{}    {} {} {} {}".format(
                time,
                rv[0],
                rv[1],
                rv[2],
                rv[3],
            ))
            logger2.info("{} {}".format(
                time,
                cpu_time() - cpu_t0,
            ))

    return


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    for t in [0, 300]:
        test_heom(
            fname=f'heom.dat',
            dof=1,
            max_tier=50,
            rank_heom=4,
            decomp_method=None,
            k_max=0,
            temperature=t,
            coupling=1750,
            ode_method='RK45',
            ps_method=None,
            scale=1.0,
        )
