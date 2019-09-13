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

from minitn.lib.tools import time_this
from minitn.lib.units import Quantity
from minitn.algorithms.ml import MultiLayer
from minitn.models.spinboson import SpinBosonModel
from minitn.tensor import Leaf, Tensor


@time_this
def sbm_ft(including_bath=False):
    # Define parameters of the model.
    sbm = SpinBosonModel(
        including_bath=including_bath,
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
    graph, root = sbm.autograph_with_aux(n_branch=2) 
    root = Tensor.generate(graph, root)

    # Define the detailed parameters for the MC-MCTDH tree
    solver = MultiLayer(root, sbm.h_list, f_list=sbm.f_list,
                        use_str_name=True)
    bond_dict = {}
    # Leaves
    for s, i, t, j in root.linkage_visitor():
        if isinstance(t, Leaf):
            dim = sbm.dimensions[t.name]
            bond_dict[(s, i, t, j)] = dim
            s_ax = s.axis
            p, p_ax = s[s_ax]
            bond_dict[(p, p_ax, s, s_ax)] = dim ** 2 if dim < 9 else 50
    # ELEC part
    elec_r = root[0][0]
    for s, i, t, j in elec_r.linkage_visitor(leaf=False):
        if (s, i, t, j) not in bond_dict:
            raise NotImplementedError()
    # INNER part
    inner_r = root[1][0] if including_bath else root
    if including_bath:
        bond_dict[(root, 1, inner_r, 0)] = 60
    for s, i, t, j in inner_r.linkage_visitor(leaf=False):
        if (s, i, t, j) not in bond_dict:
            bond_dict[(s, i, t, j)] = 50
    # OUTER part
    if including_bath:
        outer_r = root[2][0]
        bond_dict[(root, 2, outer_r, 0)] = 20
        for s, i, t, j in root[2][0].linkage_visitor(leaf=False):
            if (s, i, t, j) not in bond_dict:
                bond_dict[(s, i, t, j)] = 10
    solver.autocomplete(bond_dict, max_entangled=True)

    # Define the computation details
    solver.settings(
        cmf_steps=10,
        ode_method='RK45',
        ps_method='split-unite'
    )
    print("Size of a wfn: {} complexes".format(len(root.vectorize())))

    # Do the imaginary time propogation
    inv_tem = 1 / 500
    steps = 1000
    for time, _ in solver.propagator(
        steps=steps,
        ode_inter=Quantity(inv_tem / steps / 2, unit='K-1').value_in_au,
        split=True,
        imaginary=True
    ):
        t = 2 * Quantity(time).convert_to(unit='K-1').value
        z = solver.relative_partition_function
        kelvin = 'inf' if abs(t) < 1.e-14 else 1.0 / t
        logging.warning('Temperatue: {} K; ln(Z/Z_0): {}'
                        .format(kelvin, np.log(z)))

    # Define the obersevable of interest
    projector = np.array([[0., 0.],
                        [0., 1.]])
    op = [[[root[0][0][1][0], projector]]]

    # Do the real time propogation
    tp_list = []
    steps=2000
    root.is_normalized=True
    for time, _ in solver.propagator(
        steps=steps,
        ode_inter=Quantity(100 / steps, 'fs').value_in_au,
        split=True,
        imaginary=False,
        move_energy=True
    ):
        t = Quantity(time).convert_to(unit='fs').value
        p = solver.expection(op=op)
        logging.warning('Time: {:.2f} fs; P2: {}'.format(t, p))
        tp_list.append((t, p))

    # Save the results
    msg = 'split-origin'
    np.save('sbm-ft-{}'.format(msg), tp_list)


logging.basicConfig(
    format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)
sbm_ft(including_bath=False)
