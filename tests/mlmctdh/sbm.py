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
from minitn.models.particles import Phonon
from minitn.tensor import Leaf, Tensor

DTYPE = np.complex128


class SpectralDensityFunction:
    """
    the sub-ohmic spectral density function
    """

    def __init__(self, alpha, omega_c, s):
        self.alpha = alpha
        self.omega_c = omega_c
        self.s = s

    def adiabatic_renormalization(self, delta, p: float):
        """
        the cut-off omega_l is p*delta
        """
        loop = 0
        re = 1.
        while loop < 50:
            re_old = re
            omega_l = delta * re * p

            def integrate_func(x):
                return self.func(x) / x**2

            res = quad(integrate_func, a=omega_l, b=self.omega_c * 30)
            logging.info(f"integrate: {res[0]}, {res[1]}")
            re = np.exp(-res[0] * 2 / np.pi)
            loop += 1
            logging.info(f"re, {re_old}, {re}")
            if np.allclose(re, re_old):
                break

        return delta * re, delta * re * p

    def func(self, omega_value):
        """
        the function of the ohmic spectral density function
        """
        return np.pi / 2. * self.alpha * omega_value**self.s *\
                self.omega_c**(1-self.s) * np.exp(-omega_value / self.omega_c)

    def _dos_Wang1(self, nb, omega_value):
        r"""
        Wang's 1st scheme DOS \rho(\omega)
        """
        return (nb + 1) / self.omega_c * np.exp(-omega_value / self.omega_c)

    def Wang1(self, nb):
        """
        Wang's 1st scheme discretization
        """
        omega_value = np.array([-np.log(-float(j) / (nb + 1) + 1.) * self.omega_c for j in range(1, nb + 1, 1)])

        # general form
        c_j2 = 2. / np.pi * omega_value * self.func(omega_value) / self._dos_Wang1(nb, omega_value)

        return omega_value, c_j2

    def trapz(self, nb, x0, x1):
        dw = (x1 - x0) / float(nb)
        xlist = [x0 + i * dw for i in range(nb + 1)]
        omega_value = np.array([(xlist[i] + xlist[i + 1]) / 2. for i in range(nb)])
        c_j2 = np.array([(self.func(xlist[i]) + self.func(xlist[i + 1])) / 2 for i in range(nb)
                        ]) * 2. / np.pi * omega_value * dw

        return omega_value, c_j2


def ml(fname, e, v, primitive_dim, spf_dim, ph_parameters, steps=2000, ode_inter=0.1):
    logger = Logger(filename=fname).logger

    # define parameters
    sys_hamiltonian = np.array([[e, v], [v, -e]], dtype=DTYPE)
    projector = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=DTYPE)  # S in H_SB = S x B

    primitive_dim = primitive_dim
    spf_dim = spf_dim

    # Define all Leaf tensors and hamiltonian we need
    h_list = []
    sys_leaf = Leaf(name='sys0')
    h_list.append([(sys_leaf, -1.0j * sys_hamiltonian)])

    ph_parameters = ph_parameters

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
        return Tensor(name='spf{}'.format(n), axis=0, normalized=True)

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
        root.normalized = True
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in graph[parent]:
            parent.link_to(parent.order, child, 0)
            if child in graph:
                stack.append(child)
    logger.info(f"graph:{graph}")

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
    logger.info(f"bond_dict:{bond_dict}")
    # set initial root array
    a, b = 1.0, 0
    init_proj = np.array([[a, 0.0], [b, 0.0]]) / np.sqrt(a**2 + b**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    # Define the computation details
    solver.ode_method = 'RK45'
    solver.snd_order = False
    solver.cmf_steps = 100
    # Define the obersevable of interest
    logger.info('''# time    rho00  rho01  rho10  rho11''')
    for time, _ in solver.propagator(
            steps=steps,
            ode_inter=ode_inter,
            split=True,
    ):
        t = time
        for tensor in root.visitor(axis=None):
            tensor.reset()
            tensor.normalize(forced=True)
        rho = root.partial_env(0, proper=False)
        for tensor in root.visitor(axis=None):
            tensor.reset()
        flat_data = [t] + list(np.reshape(rho, -1))
        logger.info('{}    {}  {}  {}  {}'.format(*flat_data))


alpha = 0.05
omega_c = 20
nmodes = 4
s = 0.5
sdf = SpectralDensityFunction(alpha, omega_c, s)
w, c2 = sdf.Wang1(nmodes)

ph_parameters = list(zip(w, np.sqrt(2 * c2 / w)))
print(ph_parameters)
e = 0
v = 0
primitive_dim = 20
spf_dim = 10

logging.basicConfig(format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s', level=logging.INFO)
ml(None, e, v, primitive_dim, spf_dim, ph_parameters, steps=2000, ode_inter=0.1)
