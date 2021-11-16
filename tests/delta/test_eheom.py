#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

from builtins import filter, map, range, zip

from minitn.lib.backend import DTYPE, np
from minitn.heom.hierachy import Hierachy
from minitn.heom.corr import Correlation
from minitn.heom.propagate import MultiLayer
from minitn.lib.logging import Logger
from minitn.lib.tools import huffman_tree
from minitn.models.bath import linear_discretization, SpectralDensityFactory
from minitn.models.sbm import SBM
from minitn.lib.units import Quantity
from minitn.tensor import Tensor, Leaf

# System: pure dephasing
e = Quantity(6500, 'cm-1').value_in_au
v = Quantity(500, 'cm-1').value_in_au
eta = Quantity(500, 'cm-1').value_in_au
omega0 = Quantity(2000, 'cm-1').value_in_au
dof = 2
max_tier = 20

ph_parameters = linear_discretization(SpectralDensityFactory.plain(eta, omega0), omega0, dof)

model = SBM(sys_ham=np.array([[-e / 2.0, v], [v, e / 2.0]], dtype=DTYPE),
            sys_op=np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE),
            ph_parameters=ph_parameters,
            n_dims=(dof * [max_tier]))

# init state
A, B = 1.0, 1.0
wfn_0 = np.array([A, B]) / np.sqrt(A**2 + B**2)
rho_0 = np.tensordot(wfn_0, wfn_0, axes=0)

# Propagation
dt_unit = Quantity(0.1, 'fs').value_in_au
callback_interval = 10
count = 10000


def test_heom(fname=None):
    n_dims = 2 * dof * [max_tier]
    root = simple_heom(rho_0, n_dims)
    leaves = root.leaves()
    h_list = model.heom_h_list(leaves[-2], leaves[-1], leaves[:-2], beta=None)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = 1000

    # Define the obersevable of interest
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
            split=True,
    )):
        if n % callback_interval == 0:
            rho = np.reshape(r.array, (-1, 4))
            t = Quantity(time).convert_to(unit='fs').value
            logger.info("{} {} {} {} {}".format(t, rho[0, 0], rho[0, 1], rho[0, 2], rho[0, 3]))
    return


def test_mctdh(fname=None):
    sys_leaf = Leaf(name='sys0')

    ph_leaves = []
    for n, (omega, g) in enumerate(ph_parameters, 1):
        ph_leaf = Leaf(name='ph{}'.format(n))
        ph_leaves.append(ph_leaf)

    def ph_spf():
        return Tensor(name='spf', axis=0)

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

    print(graph)
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
                bond_dict[(s, i, t, j)] = max_tier // 2
    solver.autocomplete(bond_dict)
    # set initial root array
    init_proj = np.array([[A, 0.0], [B, 0.0]]) / np.sqrt(A**2 + B**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    solver = MultiLayer(root, h_list)
    solver.ode_method = 'RK45'
    solver.cmf_steps = 1000

    # Define the obersevable of interest
    logger = Logger(filename=fname, level='info').logger
    for n, (time, r) in enumerate(solver.propagator(
            steps=count,
            ode_inter=dt_unit,
            split=True,
    )):
        if n % callback_interval == 0:
            a = root.array
            rho = Tensor.partial_trace(a, 0, a, 0)
            t = Quantity(time).convert_to(unit='fs').value
            logger.info("{}    {} {} {} {}".format(t, rho[0, 0], rho[0, 1], rho[1, 0], rho[1, 1]))


if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt

    f_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(os.path.join(f_dir, 'data'))
    prefix = "Delta-{}".format(max_tier)

    test_heom(fname='{}_heom.dat'.format(prefix))

    # test_mctdh(fname='{}_wfn.dat'.format(prefix))
