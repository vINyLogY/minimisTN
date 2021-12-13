#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from minitn.heom.corr import Drude

from minitn.heom.network import tensor_train_template
from minitn.heom.propagate import MultiLayer
from minitn.lib.backend import DTYPE, np
from minitn.lib.logging import Logger
from minitn.lib.tools import huffman_tree
from minitn.lib.units import Quantity
from minitn.models.sbm import SBM
from minitn.tensor import Leaf, Tensor

# System: pure dephasing
e = Quantity(5000, 'cm-1').value_in_au
v = Quantity(500, 'cm-1').value_in_au

max_tier = 15
rank_heom = max_tier
rank_wfn = max_tier
beta = Quantity(1 / 300, 'K-1').value_in_au
#beta = None


ph_parameters = [
    (Quantity(400, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    (Quantity(800, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    (Quantity(1200, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    (Quantity(1600, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
]
dof = len(ph_parameters)
prefix = 'boson_fk_type3_dof{}_ZT_t{}_'.format(dof, max_tier)

drude = Drude(
    gamma=Quantity(20, 'cm-1').value_in_au,
    lambda_=Quantity(400, 'cm-1').value_in_au,
    beta=beta,
)

model = SBM(
    sys_ham=np.array([[-0.5 * e, v], [v, 0.5 * e]], dtype=DTYPE),
    sys_op=np.array([[-0.5, 0.0], [0.0, 0.5]], dtype=DTYPE),
    ph_parameters=ph_parameters,
    ph_dims=(dof * [max_tier]),
    #bath_corr=drude,
    #bath_dims=[max_tier],
)

# init state
A, B = 1.0, 1.0
wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

# Propagation
dt_unit = Quantity(0.01, 'fs').value_in_au
callback_interval = 10
count = 100_000


def test_heom(fname=None):
    ph_dims = list(np.repeat(model.ph_dims, 2))
    n_dims = ph_dims if model.bath_dims is None else ph_dims + model.bath_dims
    print(n_dims)

    root = tensor_train_template(rho_0, n_dims, rank=rank_heom)
    leaves = root.leaves()
    h_list = model.heom_h_list(leaves[0], leaves[1], leaves[2:], beta=beta)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # use constant mean-field
    solver.ps_method = 'split'
    #solver.svd_err = 1.0e-14

    # Define the obersevable of interest
    logger = Logger(filename=prefix + fname, level='info').logger
    logger2 = Logger(filename=prefix + "en_" + fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
            split=True,
    )):
        # renormalized by the trace of rho
        norm = np.trace(np.reshape(np.reshape(r.array, (4, -1))[:, 0], (2, 2)))
        r.set_array(r.array / norm)
        if n % callback_interval == 0:
            t = Quantity(time).convert_to(unit='fs').value
            rho = np.reshape(r.array, (4, -1))[:, 0]
            logger.info("{}    {} {} {} {}".format(t, rho[0], rho[1], rho[2], rho[3]))
            en = np.trace(np.reshape(rho, (2, 2)) @ model.h)
            logger2.info('{}    {}'.format(t, en))
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
                bond_dict[(s, i, t, j)] = rank_wfn
    solver.autocomplete(bond_dict)
    # set initial root array
    init_proj = np.array([[A, 0.0], [B, 0.0]]) / np.sqrt(A**2 + B**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # constant mean-field
    solver.ps_method = 'split'
    solver.svd_err = 1.0e-14

    # Define the obersevable of interest
    logger = Logger(filename=prefix + fname, level='info').logger
    logger2 = Logger(filename=prefix + 'en_' + fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
            split=True,
    )):
        if n % callback_interval == 0:
            t = Quantity(time).convert_to(unit='fs').value
            rho = r.partial_env(0, proper=False)
            logger.info("{}    {} {} {} {}".format(t, rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]))
            en = np.trace(rho @ model.h)
            logger2.info('{}    {}'.format(t, en))


if __name__ == '__main__':
    import os
    import sys

    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'data'))

    test_heom(fname='heom.dat')
    #test_mctdh(fname='wfn.dat')
