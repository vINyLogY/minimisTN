#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from time import time as cpu_time

from minitn.heom.corr import Drude

from minitn.heom.network import simple_heom, tensor_train_template
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.units import Quantity
from minitn.models.sbm import SpinBoson


def test_heom(
    fname=None,
    dof=3,
    max_tier=10,
    k_max=6,
    decomp_method=None,
    f_type=0,
):
    # System:
    e = Quantity(5000, 'cm-1').value_in_au
    relaxed = False
    v = Quantity(500, 'cm-1').value_in_au if relaxed else 0.0

    rank_heom = max_tier**2 // 2 if decomp_method is not None else None
    ps_method = 'split'

    temperature = 300
    beta = Quantity(1 /
                    temperature, 'K-1').value_in_au if temperature else None

    ph_parameters = [
        (Quantity(1600, 'cm-1').value_in_au, Quantity(500,
                                                      'cm-1').value_in_au),
        (Quantity(1800, 'cm-1').value_in_au, Quantity(500,
                                                      'cm-1').value_in_au),
        (Quantity(1400, 'cm-1').value_in_au, Quantity(500,
                                                      'cm-1').value_in_au),
        (Quantity(2000, 'cm-1').value_in_au, Quantity(500,
                                                      'cm-1').value_in_au),
    ][:dof]

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
    dt_unit = Quantity(1., 'fs').value_in_au
    callback_interval = 1
    count = 10

    prefix = (
        f'boson-drude_perf_{"relaxed" if relaxed else "pure"}_'
        f'{decomp_method}_dof{dof}_bcf{k_max}_t{max_tier}_r{rank_heom}_{temperature}K_{ps_method}'
    )
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
                               f_type=f_type)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    solver.ps_method = ps_method
    #solver.svd_err = 1.0e-14  #only for unite propagation

    # Define the obersevable of interest
    cpu_t0 = cpu_time()
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(
            solver.propagator(
                steps=count,
                ode_inter=dt_unit,
                split=True,
            )):
        # renormalized by the trace of rho
        norm = np.trace(np.reshape(np.reshape(r.array, (4, -1))[:, 0], (2, 2)))
        r.set_array(r.array / norm)
        if n % callback_interval == 0:
            rho = np.reshape(r.array, (4, -1))[:, 0]
            logger.info("{}    {} {} {} {}    {}".format(
                time,
                rho[0],
                rho[1],
                rho[2],
                rho[3],
                cpu_time() - cpu_t0,
            ))

    return


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(f_dir)

    # for dof in [1, 2, 3]:
    #     for bcf_term in [2, 5, 8]:
    #         for depth in [6, 9, 12, 15]:
    #             test_heom(
    #                 fname='heom.dat',
    #                 dof=dof,
    #                 decomp_method='TT',
    #                 k_max=bcf_term,
    #                 max_tier=depth,
    #             )

    for dof in [1]:
        for bcf_term in [5]:
            for depth in [12]:
                try:
                    test_heom(
                        fname='heom.dat',
                        dof=dof,
                        decomp_method=None,
                        k_max=bcf_term,
                        max_tier=depth,
                    )
                except:
                    continue
