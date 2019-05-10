#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division


import logging
from builtins import filter, map, range, zip
from functools import partial
from itertools import count

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from minitn.lib.tools import __, time_this, figure
from ft_sho_model import test_2layers


@time_this
def main():
    n_dvr = 50
    dofs = 2
    exp = test_2layers(n_spf=10, n_dvr=n_dvr, dofs=dofs)
    exp.settings(cmf_steps=1,
                 ode_method='RK23',
                 ps_method='s')
    p = exp.propagator(ode_inter=0.01, split=True, imaginary=True)
    for (t1, _), (t, z) in zip(p, ref(n_dvr=n_dvr, dofs=dofs)):
        assert abs(2 * t1 - t) < 1.e-8
        z1 = exp.relative_partition_function * (n_dvr ** dofs)
        msg = "beta:{:.4f}   Z1:{:.8f}   Z:{:.8f}".format(
            t, z1, z
        )
        print(msg)
    return


def ref(ode_inter=0.01, c=0.5, n_dvr=40, dofs=2):
    a = c ** 2 
    po = np.identity(dofs)
    po += a * np.eye(dofs, k=1)
    po += a * np.eye(dofs, k=(-1))
    w = np.sqrt(linalg.eigh(po, eigvals_only=True))
    for n in count():
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
        yield (beta, z)


logging.basicConfig(
    format='%(levelname)s: (In %(funcName)s, %(module)s)  %(message)s',
    level=logging.WARNING
)
main()
