#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from minitn.heom.corr import Brownian

from minitn.heom.network import simple_heom
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.tools import huffman_tree
from minitn.lib.units import Quantity
from minitn.models.sbm import SpinBoson
from minitn.tensor import Leaf, Tensor
import os, sys

from scipy import linalg

# System:
e = 0.0
v = 1.0
max_tier = 15
rank_heom = max_tier
ps_method = 'split'
temperature = 'FT'
beta = 1.0 if temperature == 'FT' else None
# beta = None: ZT

ph_parameters = []
k_max = 0

lambda_ = 2.0
omega = 0.5
gamma = float(sys.argv[1])

dof = len(ph_parameters)
prefix = f'boson_direct_l{lambda_}_o{omega}_g{gamma}_t{max_tier}'

bath = Brownian(
    lambda_=2.0,
    omega=omega,
    gamma=gamma,
    k_max=0,
    beta=beta,
)

model = SpinBoson(
    sys_ham=np.array([[-0.5 * e, v], [v, 0.5 * e]], dtype=DTYPE),
    sys_op=np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=DTYPE),
    ph_parameters=ph_parameters,
    ph_dims=(dof * [max_tier]),
    bath_corr=bath,
    bath_dims=[max_tier] * (2 + k_max),
)

# init state
A, B = 1.0, 0.0
wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

# Propagation
dt_unit = 0.01
callback_interval = 10
count = 30000


def test_diag(fname=None, f_type=0):
    fname = prefix + '_f{}_'.format(f_type) + fname
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    root = simple_heom(rho_0, n_dims)
    leaves = root.leaves()
    ham = model.dense_h(leaves[0],
                        leaves[1], [],
                        bath_indices=leaves[2:],
                        beta=beta,
                        f_type=f_type)

    w = linalg.eigvals(ham)
    np.savetxt(f"{fname}", w)

    return


def test_heom(fname=None, f_type=0):
    fname = prefix + '_f{}_'.format(f_type) + fname
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    root = simple_heom(rho_0, n_dims)
    leaves = root.leaves()
    h_list = model.heom_h_list(leaves[0],
                               leaves[1], [],
                               bath_indices=leaves[2:],
                               beta=beta,
                               f_type=f_type)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    solver.ps_method = ps_method
    #solver.svd_err = 1.0e-14

    # Define the obersevable of interest
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
            logger.info("{}    {} {} {} {}".format(time, rho[0], rho[1],
                                                   rho[2], rho[3]))

    return


if __name__ == '__main__':
    import os

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, '2022data', 'brownian'))

    test_diag(fname='brownian_spec.dat', f_type=0)
    test_heom(fname='brownian_heom.dat', f_type=0)
