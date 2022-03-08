#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from minitn.heom.corr import Drude

from minitn.heom.network import simple_heom, tensor_train_template
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.tools import huffman_tree
from minitn.lib.units import Quantity
from minitn.models.sbm import SpinBoson
from minitn.tensor import Leaf, Tensor

from scipy import linalg
import sys

# System:
e = Quantity(5000, 'cm-1').value_in_au
v = 0.0
max_tier = 10
rank_heom = max_tier
wfn_rank = max_tier
ps_method = 'split'
decomp_method = None
temperature = 300
beta = Quantity(1 / temperature, 'K-1').value_in_au if temperature else None
# beta = None: ZT

ph_parameters = [
    #(Quantity(1600, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
]
dof = len(ph_parameters)

k_max = int(sys.argv[1])
drude = Drude(
    gamma=Quantity(20, 'cm-1').value_in_au,
    lambda_=Quantity(500, 'cm-1').value_in_au,
    beta=beta,
    k_max=k_max,
)

model = SpinBoson(
    sys_ham=np.array([[0.0, v], [v, e]], dtype=DTYPE),
    sys_op=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
    ph_parameters=ph_parameters,
    ph_dims=(dof * [max_tier]),
    bath_corr=drude,
    bath_dims=[max_tier] * k_max,
)

# init state
A, B = 1.0, 1.0
wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

# Propagation
dt_unit = Quantity(0.01, 'fs').value_in_au
callback_interval = 10
count = 1000_00

prefix = f'bosondrude_{decomp_method}_dof{dof}_bcf{k_max}_{temperature}K_t{max_tier}_{ps_method}_'

#prefix = f'boson_{decomp_method}_dof{dof}_bcf{k_max}_{temperature}K_t{max_tier}_{ps_method}_'

#prefix = f'drude_{decomp_method}_dof{dof}_bcf{k_max}_{temperature}K_t{max_tier}_{ps_method}_'


def test_diag(fname=None, f_type=0):
    fname = 'type{}'.format(f_type) + fname
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    root = simple_heom(rho_0, n_dims)
    #root = tensor_train_template(rho_0, n_dims, rank=max_tier)
    leaves = root.leaves()
    ham = model.dense_h(leaves[0],
                        leaves[1],
                        leaves[2:],
                        beta=beta,
                        f_type=f_type)

    w = linalg.eigvals(ham)
    np.savetxt(f"{fname}-spec.txt", w)

    return


def test_heom(fname=None, f_type=0):
    fname = prefix + 'f{}_'.format(f_type) + fname
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    if decomp_method is None:
        root = simple_heom(rho_0, n_dims)
    elif decomp_method == 'TT':
        root = tensor_train_template(rho_0, n_dims, rank=max_tier)
    else:
        raise NotImplementedError
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
    #solver.svd_err = 1.0e-14  #only for unite propagation

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


def test_mctdh(fname=None):
    assert beta is None
    sys_leaf = Leaf(name='sys0')

    ph_leaves = []
    for n, (omega, g) in enumerate(ph_parameters, 1):
        ph_leaf = Leaf(name='ph{}'.format(n))
        ph_leaves.append(ph_leaf)

    def ph_spf():
        t = Tensor(axis=0)
        t.name = 'spf' + str(hex(id(t)))[-4:]
        return t

    graph, root = huffman_tree(ph_leaves, obj_new=ph_spf, n_branch=2)
    try:
        graph[root].insert(0, sys_leaf)
    except KeyError:
        ph_leaf = root
        root = Tensor()
        graph[root] = [sys_leaf, ph_leaf]
    finally:
        root.name = 'wfn'
        root.axis = None

    stack = [root]
    while stack:
        parent = stack.pop()
        for child in graph[parent]:
            parent.link_to(parent.order, child, 0)
            if child in graph:
                stack.append(child)

    # Define the detailed parameters for the ML-MCTDH tree
    h_list = model.wfn_h_list(sys_leaf, ph_leaves)
    solver = MultiLayer(root, h_list)
    bond_dict = {}
    # Leaves
    for s, i, t, j in root.linkage_visitor():
        if t.name.startswith('sys'):
            bond_dict[(s, i, t, j)] = 2
        else:
            if isinstance(t, Leaf):
                bond_dict[(s, i, t, j)] = max_tier
            else:
                bond_dict[(s, i, t, j)] = wfn_rank
    solver.autocomplete(bond_dict)
    # set initial root array
    init_proj = np.array([[A, 0.0], [B, 0.0]]) / np.sqrt(A**2 + B**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # constant mean-field
    solver.ps_method = ps_method
    solver.svd_err = 1.0e-14

    # Define the obersevable of interest
    logger = Logger(filename=prefix + fname, level='info').logger
    logger2 = Logger(filename=prefix + 'en_' + fname, level='info').logger
    for n, (time, r) in enumerate(
            solver.propagator(
                steps=count,
                ode_inter=dt_unit,
                split=True,
            )):
        if n % callback_interval == 0:
            t = Quantity(time).convert_to(unit='fs').value
            rho = r.partial_env(0, proper=False)
            logger.info("{}    {} {} {} {}".format(t, rho[0, 0], rho[0, 1],
                                                   rho[1, 0], rho[1, 1]))
            en = np.trace(rho @ model.h)
            logger2.info('{}    {}'.format(t, en))


if __name__ == '__main__':
    import os

    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, '2022data', 'puredephasing'))

    for f_type in [
            0,
    ]:
        #test_diag(fname='heom.dat', f_type=f_type)
        test_heom(fname='heom.dat', f_type=f_type)
