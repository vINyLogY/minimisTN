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
# import json

import numpy as np
from scipy import linalg

from minitn.lib.tools import time_this
from minitn.lib.units import Quantity
from minitn.algorithms.ml import MultiLayer
from minitn.models.spinboson import SpinBosonModel
from minitn.tensor import Leaf, Tensor


@time_this
def sbm_zt(omega, lambda_, split=False, snd=False):
    sbm = SpinBosonModel(
        including_bath=False,
        e1=0.,
        e2=Quantity(6500, 'cm-1').value_in_au,
        v=Quantity(500, 'cm-1').value_in_au,
        omega_list=[
            Quantity(omega, 'cm-1').value_in_au
        ],
        lambda_list=[
            Quantity(lambda_, 'cm-1').value_in_au
        ],
        dim_list=[20, 20, 20, 20],
        # reserved for laser
        omega=13000,
        mu=Quantity(250, 'cm-1').value_in_au,
        tau=Quantity(30, 'fs').value_in_au,
        t_d=Quantity(60, 'fs').value_in_au,
    )

    # Define the topological structure of the ML-MCTDH tree
    graph, root = {
        'ROOT': [sbm.elec_leaf] + sbm.inner_leaves,
    }, 'ROOT'
    root = Tensor.generate(graph, root)

    # Define the detailed parameters for the MC-MCTDH tree
    solver = MultiLayer(root, sbm.h_list, # f_list=sbm.f_list,
                        use_str_name=True)
    bond_dict = {}
    # Leaves
    for s, i, t, j in root.linkage_visitor():
        if isinstance(t, Leaf):
            bond_dict[(s, i, t, j)] = sbm.dimensions[t.name]

    solver.autocomplete(bond_dict, max_entangled=False)
    # manually set root array
    init_proj = np.array([
        [1.0, 0.0],
        [1.0, 0.0]
    ]) / np.sqrt(2.0)
    root_array = Tensor.partial_product(root.array, 0, init_proj, 1)
    root.set_array(root_array)
    # Define the computation details
    solver.settings(
        max_ode_steps=100,
        cmf_steps=(1 if split else 10),
        ode_method='RK45',
        ps_method='s',
        snd_order=snd,
    )
    root.is_normalized=True
    # Define the obersevable of interest
    projector = np.array([[0., 0.],
                        [0., 1.]])
    op = [[[root[0][0], projector]]]
    t_p = []
    for time, _ in solver.propagator(
        steps=2000,
        ode_inter=Quantity(0.05, 'fs').value_in_au,
        split=split,
        move_energy=True,
    ):
        t, p = (Quantity(time).convert_to(unit='fs').value,
                solver.expection(op=op))
        for tensor in root.visitor(axis=None):
            tensor.reset()
        rho = root.partial_env(0, proper=False)
        for tensor in root.visitor(axis=None):
            tensor.reset()
        w, _ = np.linalg.eigh(rho)
        purity = np.sum(w**2)
        #entropy = np.sum(w * np.log(w))
        t_p.append((t, np.abs(p), -np.abs(purity)))
        logging.warning('Time: {:.2f} fs, P2: {}, purity: {}'.format(t, p, purity))

    # Save the results
    msg = 'split' if split else 'origin'
    msg2 = 'snd' if snd else 'fst'
    np.savetxt('1_{}_const-coupling_time-pop-purity_{}-{}.txt'.format(int(omega), msg, msg2), t_p)


def raw_interpolation(omega, lambda_):
    # using a.u.
    omega = np.array([Quantity(w, 'cm-1').value_in_au for w in omega])
    lambda_ = np.array([Quantity(l, 'cm-1').value_in_au for l in lambda_])
    coupling = np.sqrt(2.0 * lambda_) * omega
    tau_inv_sq = np.sum(coupling ** 2 / (4.0 * omega))
    t_space = np.linspace(0, Quantity(100, 'fs').value_in_au, 1000)
    pr_space = 0.5 + 0.5 * np.exp(-2.0 * (t_space**2) * tau_inv_sq)
    p_space = t_space * 0.0
    t_space = list(map(lambda x: Quantity(x).convert_to('fs').value, t_space))
    t_p = np.array(list(zip(t_space, p_space, pr_space)))
    np.savetxt('1_formula.txt', t_p)
    return


logging.basicConfig(
    format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)
# unit in cm^{-1}
omega = np.array([2100., 650., 400., 150.])
lambda_ = np.array([750., 750., 750., 750.])
coupling= np.sqrt(0.5 * lambda_ * (omega ** 2))
eff_omega=np.sqrt(np.dot(coupling ** 2, omega ** 2) / np.dot(coupling, coupling))
eff_lambda=np.dot(coupling, coupling) * 2.0 / eff_omega ** 2
raw_interpolation(omega, lambda_)
#sbm_zt(eff_omega, eff_lambda, split=True, snd=True)
