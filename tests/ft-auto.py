#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division


import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from minitn.lib.tools import __, time_this, figure
from ft_sho_model import test_2layers


@time_this
def main():
    exp = test_2layers()
    exp.settings(ode_method='RK23',
                 ps_method='s')
    p1 = exp.propagator(ode_inter=0.1, imaginary=True)
    exp2 = test_2layers()
    exp2.settings(cmf_steps=10,
                  ode_method='RK23')
    p2 = exp2.propagator(ode_inter=0.1, imaginary=True)
    for (t1, r1), (t2, r2), (t, v) in zip(p1, p2, ref()):
        v1, v2 = (r1.global_norm()) ** 2, (r2.global_norm()) ** 2
        msg = "beta:{} v1:{} v2:{} v:{}".format(t, v1, v2, v)
        print(msg)
    return


def ref(ode_inter=0.1, c=0.5, n_dvr=40, dofs=2):
    po = np.identity(dofs)
    po += c * np.eye(dofs, k=1)
    po += c * np.eye(dofs, k=(-1))
    w = np.sqrt(linalg.eigh(po, eigvals_only=True))
    for n in range(100):
        beta = 2 * n * ode_inter
        exp = np.exp

        def _z(w):
            ans = 0.0
            for i in range(n_dvr):
                ans += exp(-beta * (i + 0.5) * w)
            return ans

        z_i = list(map(_z, w))
        z = np.prod(z_i)
        logging.info('beta: {:.3f}; Z: {:.8f}'.format(beta, z))
        yield (beta / 2, z / (n_dvr ** dofs))


logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s',
    level=logging.DEBUG+1
)
main()
