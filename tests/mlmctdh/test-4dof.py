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


from model import ml
import logging

logging.basicConfig(
    format='%(asctime)s-%(levelname)s: (In %(module)s)[%(funcName)s] %(message)s',
    level=logging.INFO
)
ml(dof=4, eta=500, cutoff=2000, loc=None, steps=2000, ode_inter=0.1)

