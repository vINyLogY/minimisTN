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
from minitn.models.sbm import SpinBoson
from minitn.tensor import Leaf, Tensor
from scipy import linalg as la

from minitn.models.particles import Phonon

# System:
e = Quantity(5000, 'cm-1').value_in_au
v = Quantity(500, 'cm-1').value_in_au
max_tier = 15
rank_heom = 1
temperature = 300
beta = Quantity(1 / temperature, 'K-1').value_in_au if temperature else None
# beta = None: ZT

ph_parameters = [
    #(Quantity(400, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    #(Quantity(800, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    #(Quantity(1200, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
    (Quantity(1600, 'cm-1').value_in_au, Quantity(500, 'cm-1').value_in_au),
]
dof = len(ph_parameters)
prefix = 'boson_dof{}_{}K_t{}_'.format(dof, temperature, max_tier)

drude = Drude(
    gamma=Quantity(20, 'cm-1').value_in_au,
    lambda_=Quantity(400, 'cm-1').value_in_au,
    beta=beta,
)

model = SpinBoson(
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
count = 50_00


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
    solver.ps_method = 'unite'
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
    n_state = 2
    sys_i = Leaf(name='sys-i')
    sys_j = Leaf(name='sys-j')
    sys_root = Tensor(name='elec', axis=0)

    rank_wfn = 1
    ph_spdos = []
    for n, (omega, g) in enumerate(ph_parameters, 1):
        ph_spdo = Tensor(name='ph{}'.format(n), axis=0)
        ph_spdos.append(ph_spdo)

    def ph_spf():
        t = Tensor(axis=0)
        t.name = 'spf' + str(hex(id(t)))[-4:]
        return t

    graph, ph_root = huffman_tree(ph_spdos, obj_new=ph_spf, n_branch=2)

    graph[sys_root] = [sys_i, sys_j]

    ph_is = []
    ph_js = []
    for n, ph_spdo in enumerate(ph_spdos, 1):
        phi_leaf = Leaf(name='phi{}'.format(n))
        phj_leaf = Leaf(name='phj{}'.format(n))
        graph[ph_spdo] = [phi_leaf, phj_leaf]
        ph_is.append(phi_leaf)
        ph_js.append(phj_leaf)

    root = Tensor(name='root', axis=None)
    graph[root] = [sys_root, ph_root]

    stack = [root]
    while stack:
        parent = stack.pop()
        for child in graph[parent]:
            parent.link_to(parent.order, child, 0)
            if child in graph:
                stack.append(child)

    # Define the detailed parameters for the ML-MCTDH tree
    h_list = model.do_l_list(sys_i, sys_j, ph_is, ph_js)
    solver = MultiLayer(root, h_list)
    bond_dict = {}
    # Leaves
    for s, i, t, j in root.linkage_visitor():
        if t.name.startswith('sys'):
            bond_dict[(s, i, t, j)] = n_state
        else:
            if isinstance(t, Leaf):
                bond_dict[(s, i, t, j)] = max_tier
            else:
                bond_dict[(s, i, t, j)] = rank_wfn
    solver.autocomplete(bond_dict)

    # set initial root array
    sys_root.set_array(np.array([rho_0], dtype=complex))
    for ph_spdo, (omega, _) in zip(ph_spdos, ph_parameters):
        if beta is None:
            # ZT
            array = np.zeros((max_tier, max_tier), dtype=complex)
            array[0, 0] = 1.0
        else:
            # FT
            h = Phonon(max_tier, omega).hamiltonian
            array = la.expm(-beta * h)
        ph_spdo.set_array(np.array([array], dtype=complex))

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = solver.max_ode_steps  # constant mean-field
    solver.ps_method = 'unite'
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
            for node in r.visitor(leaf=False):
                print("{}, {}".format(node, node.shape))


if __name__ == '__main__':
    import os
    import sys

    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, '2022data'))

    #test_heom(fname='heom.dat')
    test_mctdh(fname='do.dat')
