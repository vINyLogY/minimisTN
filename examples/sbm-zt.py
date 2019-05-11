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

import numpy as np
from scipy import linalg

from minitn.lib.units import Quantity
from minitn.ml import MultiLayer
from minitn.models.spinboson import SpinBosonModel
from minitn.tensor import Leaf, Tensor

logging.basicConfig(
    format='%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)

# Define parameters of the model.
sbm = SpinBosonModel(
    e1=0.,
    e2=Quantity(6500, 'cm-1').value_in_au,
    v=Quantity(500, 'cm-1').value_in_au,
    omega_list=[Quantity(2100, 'cm-1').value_in_au,
                Quantity(650, 'cm-1').value_in_au,
                Quantity(400, 'cm-1').value_in_au,
                Quantity(150, 'cm-1').value_in_au],
    lambda_list=([Quantity(750, 'cm-1').value_in_au] * 4),
    dim_list=[10, 14, 20, 30],
    stop=Quantity(10000, 'cm-1').value_in_au,
    n=32,
    dim=30,
    lambda_g=Quantity(2250, 'cm-1').value_in_au,
    omega_g=Quantity(500, 'cm-1').value_in_au,
    lambda_d=Quantity(1250, 'cm-1').value_in_au,
    omega_d=Quantity(50, 'cm-1').value_in_au,
    mu=Quantity(250, 'cm-1').value_in_au,
    tau=Quantity(30, 'fs').value_in_au,
    t_d=Quantity(60, 'fs').value_in_au,
    omega=Quantity(13000, 'cm-1').value_in_au,
)

# Define the topological structure of the ML-MCTDH tree
graph, root = sbm.autograph(n_branch=2) 
root = Tensor.generate(graph, root)

# Define the detailed parameters for the MC-MCTDH tree
solver = MultiLayer(root, sbm.h_list, f_list=sbm.f_list,
                    use_str_name=True)
bond_dict = {}
## Leaves
for s, i, t, j in root.linkage_visitor():
    if isinstance(t, Leaf):
        bond_dict[(s, i, t, j)] = sbm.dimensions[t.name]
## ELEC part
elec_r = root[0][0]
for s, i, t, j in elec_r.linkage_visitor(leaf=False):
    raise NotImplementedError()
## INNER part
inner_r = root[1][0]
bond_dict[(root, 1, inner_r, 0)] = 30
for s, i, t, j in inner_r.linkage_visitor(leaf=False):
    bond_dict[(s, i, t, j)] = 20
## OUTER part
outer_r = root[2][0]
bond_dict[(root, 2, outer_r, 0)] = 20
for s, i, t, j in root[2][0].linkage_visitor(leaf=False):
    bond_dict[(s, i, t, j)] = 10
solver.autocomplete(bond_dict, max_entangled=False)
## Following steps are not needed: phi_1 is defined as electronic
# groundstate
# Make the electron at the eigenstate of its local hamiltonian
# leaf, h = solver.h_list[0][0]
# assert leaf.name == 'ELEC'
# _, v = linalg.eigh(h)
# array = Tensor.partial_product(root.array, 0, np.transpose(v))
# root.set_array(array)

# Define the computation details
solver.settings(
    ode_method='RK45',
    ps_method='split-unite'
)

# Define the obersevable of interest
projector = np.array([[0., 0.],
                      [0., 1.]])
op=[[[elec_r, projector]]]

# Do the propogation
tp_list = []
print("Size of a wfn: {} complexes".format(len(root.vectorize())))
for time, _ in solver.propagator(
    steps=400,
    ode_inter=Quantity(0.25, 'fs').value_in_au,
    split=True,
    imaginary=False
):
    t, p = (Quantity(time).convert_to(unit='fs').value,
            solver.expection(op=op))
    logging.warning('Time: {} fs; P2: {}'.format(t, p))
    tp_list.append((t, p))

# Save the results
np.save('sbm-zt', tp_list)
