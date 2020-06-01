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
from builtins import filter, map, range, zip
from itertools import count, filterfalse

import numpy as np
from scipy import linalg
from scipy.integrate import quad

from minitn.algorithms.ml import MultiLayer
from minitn.lib.tools import time_this
from minitn.lib.units import Quantity
from minitn.models import bath
from minitn.models.particles import Phonon
from minitn.models.spinboson import SpinBosonModel
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

@time_this
def main(cutoff=1000):
    # define parameters
    e = Quantity(6500, 'cm-1').value_in_au
    v = 0.0
    eta = Quantity(250, 'cm-1').value_in_au**2
    omega0 = Quantity(cutoff, 'cm-1').value_in_au
    primitive_dim = 100
    spf_dim = 20
    dof = 3
    
    def spec_func(omega):
        if 0 < omega < omega0:
            return eta
        else:
            return 0.0
            
    # Define all Leaf tensors and hamiltonian we need
    root = Tensor(name='wfn', normalized=True)
    h_list = []
    sys_leaf = Leaf(name='sys0')
    sys_hamiltonian = np.array([[0.0, v], [v, e]], dtype=DTYPE)
    projector = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    root.link_to(0, sys_leaf, 0)
    h_list.append([(sys_leaf, sys_hamiltonian)])

    ph_parameters = linear_discretization(spec_func, omega0, dof)
    for n, (omega, g) in enumerate(ph_parameters, 1):
        ph = Phonon(primitive_dim, omega)
        ph_leaf = Leaf(name='ph{}'.format(n))
        ph_spf = Tensor(name='spf{}'.format(n), axis=0)
        root.link_to(n, ph_spf, 0)
        ph_spf.link_to(1, ph_leaf, 0)
        # hamiltonian ph part
        h_list.append([(ph_leaf, ph.hamiltonian)])
        # e-ph part
        op = ph.annihilation_operator + ph.creation_operator
        h_list.append([(ph_leaf, g * op),
                       (sys_leaf, projector)])

    # Define the detailed parameters for the MC-MCTDH tree
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
    init_proj = np.array([
        [a, 0.0],
        [b, 0.0]
    ]) / np.sqrt(a**2 + b**2)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)

    # Define the computation details
    solver.settings(
        max_ode_steps=100,
        cmf_steps=1,
        ode_method='RK45',
        ps_method='s',
        snd_order=True,
    )
    root.is_normalized=True
    # Define the obersevable of interest
    data_list = []
    for time, _ in solver.propagator(
        steps=1000,
        ode_inter=Quantity(0.1, 'fs').value_in_au,
        split=True,
        move_energy=True,
    ):
        t = Quantity(time).convert_to(unit='fs').value
        for tensor in root.visitor(axis=None):
            tensor.reset()
        rho = root.partial_env(0, proper=False)
        for tensor in root.visitor(axis=None):
            tensor.reset()
        data_list.append([t] + list(np.reshape(rho, -1)))
        logging.warning('Time: {:.2f} fs, rho: {}'.format(t, np.reshape(rho, -1)))

    # Save the results
    np.savetxt('data-{}cutoff.txt'.format(cutoff), data_list,
        header='time/fs    rho00    rho01    rho10    rho11'
    )


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
        level=logging.INFO
    )
    for cutoff in [1000, 2000, 3000, 4000, 5000]:
        main(cutoff=cutoff)
