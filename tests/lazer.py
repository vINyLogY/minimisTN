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
from minitn.ml import MultiLayer
from minitn.models.spinboson import SpinBosonModel
from minitn.tensor import Leaf, Tensor
from minitn.lib.tools import plt, figure, BraceMessage as __


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
    f = sbm.td_electron_hamiltionian()
    t_space = np.linspace(0, 100, 1000)
    tau = map(lambda x: Quantity(x, 'fs').value_in_au, t_space)
    fau = map(lambda x: f(x)[0, 1], tau)
    f_space = list(map(lambda x: Quantity(x).convert_to('cm-1').value, fau))
    with figure():
        plt.plot(t_space, f_space, 'k-')
        plt.xlim(0, 100)
        plt.xlabel(r'$t$ (fs)')
        plt.ylabel(r'$E(t)\mu~(\mathrm{cm}^{-1})$')
        plt.show()



logging.basicConfig(
    format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)
sbm_ft(including_bath=False)
