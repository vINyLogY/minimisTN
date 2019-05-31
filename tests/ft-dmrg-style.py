#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division


import logging
from builtins import filter, map, range, zip
from functools import partial
from itertools import count, product

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from minitn.lib.tools import __, time_this, figure, plt
from ft_sho_model import test_2layers


@time_this
def main():
    n_dvr = 50
    dofs = 4
    ode_inter = 0.01
    cache = []
    for m, ode_inter in product(range(3, 6), [0.005, 0.01, 0.02]):
        exp = test_2layers(n_spf=m, n_dvr=n_dvr, dofs=dofs)
        exp.settings(
            cmf_steps=1,
            ode_method='RK45',
            ps_method='s'
        )
        p = exp.propagator(ode_inter=ode_inter, split=True, imaginary=True)
        zipped = []
        for t1, _ in p:
            z1 = exp.relative_partition_function * (n_dvr ** dofs)
            zipped.append((2 * t1, np.log(np.real(z1))))
            print(m, 2 * t1, np.log(np.real(z1)))
            if 2 * t1 > 1:
                break
        cache.append((m, ode_inter, zipped))
    ref_zipped = []
    for t, z in ref(ode_inter=0.005, n_dvr=n_dvr, dofs=dofs):
        ref_zipped.append((t, np.log(z)))
        if t > 1:
            break
    with figure():
        for m, ode_inter, zipped in cache:
            t, z = zip(*zipped)
            label = "$n={}$, $\\tau={}$".format(m, ode_inter)
            plt.plot(t, z, '.', label=label)
        t, z = zip(*ref_zipped)
        plt.plot(t, z, 'k-', label='Ref')
        plt.legend(loc='best')
        plt.xlabel(r'$\beta$ (a. u.)')
        plt.ylabel(r'$\ln Z$')
        plt.show()
    np.save('tmp', cache)
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
    level=logging.WARN
)
main()
