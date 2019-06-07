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

from minitn.lib.tools import time_this
from minitn.lib.units import Quantity
from minitn.ml import MultiLayer
from minitn.models.spinboson import SpinBosonModel
from minitn.tensor import Leaf, Tensor


@time_this
def sbm_zt(including_bath=False, split=False):
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
        stop=Quantity(3 * 2250, 'cm-1').value_in_au,
        n=32,
        dim=30,
        lambda_g=Quantity(2250, 'cm-1').value_in_au,
        omega_g=Quantity(500, 'cm-1').value_in_au,
        lambda_d=Quantity(1250, 'cm-1').value_in_au,
        omega_d=Quantity(50, 'cm-1').value_in_au,
        mu=Quantity(250, 'cm-1').value_in_au,
        tau=Quantity(10, 'fs').value_in_au,
        t_d=Quantity(10, 'fs').value_in_au,
        omega=Quantity(13000, 'cm-1').value_in_au,
    )

    # Define the topological structure of the ML-MCTDH tree
    graph, root = sbm.autograph_full()
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
    solver.autocomplete(bond_dict, max_entangled=False)

    # Define the computation details
    solver.settings(
        max_ode_steps=100,
        cmf_steps=(1 if split else 10),
        ode_method='RK45',
        ps_method='split-unite'
    )
    print("Size of a wfn: {} complexes".format(len(root.vectorize())))
    root.is_normalized=True
    # Define the obersevable of interest
    projector = np.array([[0., 0.],
                        [0., 1.]])
    op = [[[root[0][0], projector]]]
    t_p = []
    for time, _ in solver.propagator(
        steps=100,
        ode_inter=Quantity(0.25, 'fs').value_in_au,
        split=split,
    ):
        t, p = (Quantity(time).convert_to(unit='fs').value,
                solver.expection(op=op))
        t_p.append((t, p))
        logging.warning('Time: {:.2f} fs, P2: {}'.format(t, p))
        if np.abs(p) > 0.5:
            break

    # Save the results
    msg = 'split' if split else 'origin'
    np.save('sbm-zt-mctdh-{}-1'.format(msg), t_p)


logging.basicConfig(
    format='%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)
sbm_zt(including_bath=False, split=False)
