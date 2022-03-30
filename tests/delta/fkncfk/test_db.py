#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from distutils.util import split_quoted
from posixpath import split
from time import time as cpu_time

from numpy import diff

from minitn.heom.corr import Drude

from minitn.heom.network import simple_heom, tensor_train_template
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.units import Quantity
from minitn.models.sbm import SpinBoson


def test_heom(fname=None,
              dof=3,
              max_tier=10,
              rank_heom=10,
              diff_type=1,
              decomp_method=None,
              ps_method='split',
              ode_method='RK45'):
    # System:
    e = Quantity(5000, 'cm-1').value_in_au
    relaxed = False
    v = Quantity(500, 'cm-1').value_in_au if relaxed else 0.0

    rank_heom = rank_heom if decomp_method is not None else None
    temperature = 0
    beta = Quantity(1 /
                    temperature, 'K-1').value_in_au if temperature else None

    ph_parameters = [
        (Quantity(1600, 'cm-1').value_in_au, Quantity(1500,
                                                      'cm-1').value_in_au),
        (Quantity(1800, 'cm-1').value_in_au, Quantity(1500,
                                                      'cm-1').value_in_au),
        (Quantity(1400, 'cm-1').value_in_au, Quantity(1500,
                                                      'cm-1').value_in_au),
        (Quantity(2000, 'cm-1').value_in_au, Quantity(1500,
                                                      'cm-1').value_in_au),
    ][:dof]

    model = SpinBoson(
        sys_ham=np.array([[0.0, v], [v, e]], dtype=DTYPE),
        sys_op=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
        ph_parameters=ph_parameters,
        ph_dims=([max_tier] * dof),
    )

    # init state
    A, B = 1.0, 1.0
    wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
    rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

    # Propagation
    dt_unit = Quantity(.01, 'fs').value_in_au
    callback_interval = 10
    count = 1000

    prefix = (
        f'boson-drude_{ode_method}_{"relaxed" if relaxed else "pure"}_'
        f'{decomp_method}_dof{dof}_t{max_tier}_r{rank_heom}_{temperature}K_{ps_method}_heom{diff_type}'
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

    if diff_type == 1:
        h_list_method = model.heom_h_list
    elif diff_type == 2:
        h_list_method = model.heom_h_list2
    else:
        raise NotImplementedError

    leaves = root.leaves()
    h_list = h_list_method(leaves[0],
                           leaves[1],
                           bath_indices=leaves[2:],
                           beta=beta)

    solver = MultiLayer(root, h_list)
    print(root.shape)
    solver.ode_method = ode_method
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    if ps_method is not None:
        solver.ps_method = ps_method
    #solver.svd_err = 1.0e-14  #only for unite propagation

    # Define the obersevable of interest
    cpu_t0 = cpu_time()
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(
            solver.propagator(
                steps=count,
                ode_inter=dt_unit,
                split=True if ps_method is not None else False,
            )):
        # renormalized by the trace of rho

        if n % callback_interval == 0:
            rho = np.trace(r.array, axis1=2, axis2=3).reshape(-1)
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

    for t in [2]:
        for depth in [10, 20, 30]:
            test_heom(
                fname='lvn.dat',
                dof=1,
                max_tier=depth,
                rank_heom=10,
                diff_type=t,
                decomp_method=None,
                ode_method='RK45',
                ps_method='split',
            )

    # for dof in [0]:
    #     for bcf_term in [5]:
    #         for depth in [6, 9, 12]:
    #             try:
    #                 test_heom(
    #                     fname='heom.dat',
    #                     dof=dof,
    #                     decomp_method=None,
    #                     k_max=bcf_term,
    #                     max_tier=depth,
    #                 )
    #             except:
    #                 continue
