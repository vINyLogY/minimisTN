#!/usr/bin/env python
# coding: utf-8
r""" A spin-boson model for photoinduced ET reactions in mixed-valence systems
in solution at zero/finite temperature.

Spin-boson model::

    H = H_e + H_v + H_B

where :math:`H_v = 1/2 \sum_j (p_j^2 + \omega_j^2(q_j -\frac{2c_j}{\omega_j^2}
|2><2|)^2)`. H_B similar (discretized from :math:`J_B`)

References
----------
.. [1] J. Chem. Phys. 124, 034114 (2006)
       https://doi.org/10.1063/1.2161178
"""
from __future__ import absolute_import, division, print_function

import logging
import os
from builtins import filter, map, range, zip
from itertools import count, filterfalse

import numpy as np
from scipy import linalg
from scipy.integrate import quad

from minitn.algorithms.ml import MultiLayer
from minitn.lib.tools import huffman_tree
from minitn.lib.logging import Logger
from minitn.lib.units import Quantity
from minitn.models.particles import Phonon
from minitn.tensor import Leaf, Tensor

DTYPE = np.complex128


def linear_discretization(spec_func, stop, num, start=0.0):
    """A simple linear method to discretize a spectral density.

    Parameters
    ----------
    spec_func : float  ->  float
        Offen denoted as J(w).
    start : float, optional
        Start point of the spectrum, defalut is 0.0.
    stop : float
        End point of the spectrum.
    num : int
        Number of modes to be given.

    Returns
    -------
    ans : [(float, float)] 
        `ans[i][0]` is the omega of one mode and `ans[i][1]` is the 
        corrospoding coupling in second quantization for all `i` in 
        `range(0, num)`.
    """

    def direct_quad(a, b):
        density = quad(spec_func, a, b)[0]
        omega = quad(lambda x: x * spec_func(x), a, b)[0] / density
        coupling = np.sqrt(density)
        return omega, coupling

    space = np.linspace(start, stop, num + 1, endpoint=True)
    omega_0, omega_1 = space[:-1], space[1:]
    ans = list(map(direct_quad, omega_0, omega_1))
    return ans


def ml(dof, e, v, eta, cutoff, scale=5, loc=None, steps=2000, ode_inter=0.1):
    f_ = 'dof{}-eta{}.log'.format(dof, eta)
    logger = Logger(filename=f_).logger

    # define parameters
    e = Quantity(e, 'cm-1').value_in_au
    v = Quantity(v, 'cm-1').value_in_au
    eta = Quantity(eta, 'cm-1').value_in_au
    omega0 = Quantity(cutoff, 'cm-1').value_in_au
    sys_hamiltonian = np.array([[-e / 2.0, v], [v, e / 2.0]], dtype=DTYPE)
    projector = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE)  # S in H_SB = S x B

    primitive_dim = 100
    spf_dim = 20

    # Spectrum function
    def spec_func(omega):
        if 0 < omega < omega0:
            return eta
        else:
            return 0.0

    # Define all Leaf tensors and hamiltonian we need
    h_list = []
    sys_leaf = Leaf(name='sys0')
    h_list.append([(sys_leaf, -1.0j * sys_hamiltonian)])

    ph_parameters = linear_discretization(spec_func, omega0, dof)
    if loc is not None:
        adj_pair = (ph_parameters[loc][0], ph_parameters[loc][1] * scale)
        ph_parameters[loc] = adj_pair
    leaves = []
    for n, (omega, g) in enumerate(ph_parameters, 1):
        ph = Phonon(primitive_dim, omega)
        ph_leaf = Leaf(name='ph{}'.format(n))
        leaves.append(ph_leaf)
        # hamiltonian ph part
        h_list.append([(ph_leaf, -1.0j * ph.hamiltonian)])
        # e-ph part
        op = ph.annihilation_operator + ph.creation_operator
        h_list.append([(ph_leaf, g * op), (sys_leaf, -1.0j * projector)])

    def ph_spf(n=0):
        n += 1
        return Tensor(name='spf{}'.format(n), axis=0)

    graph, root = huffman_tree(leaves, obj_new=ph_spf, n_branch=2)
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
    solver = MultiLayer(root, h_list)
    bond_dict = {}
    # Leaves
    for s, i, t, j in root.linkage_visitor():
        if t.name.startswith('sys'):
            bond_dict[(s, i, t, j)] = 2
        else:
            if isinstance(t, Leaf):
                bond_dict[(s, i, t, j)] = primitive_dim
            else:
                bond_dict[(s, i, t, j)] = spf_dim
    solver.autocomplete(bond_dict)
    # set initial root array
    a, b = 1.0, 1.0
    init_proj = np.array([[a, 0.0], [b, 0.0]]) / np.sqrt(a**2 + b**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    # Define the computation details
    solver.ode_method = 'RK45'
    solver.snd_order = True
    solver.cmf_steps = 1
    root.is_normalized = True
    # Define the obersevable of interest
    logger.info('''# time/fs    rho00  rho01  rho10  rho11''')
    for time, _ in solver.propagator(
            steps=steps,
            ode_inter=Quantity(ode_inter, 'fs').value_in_au,
            split=True,
    ):
        t = Quantity(time).convert_to(unit='fs').value
        for tensor in root.visitor(axis=None):
            tensor.reset()
        rho = root.partial_env(0, proper=False)
        for tensor in root.visitor(axis=None):
            tensor.reset()
        flat_data = [t] + list(np.reshape(rho, -1))
        logger.info('{}    {}  {}  {}  {}'.format(*flat_data))
