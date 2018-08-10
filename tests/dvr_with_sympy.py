#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging

import numpy as np
import sympy as sym

if __name__ == '__main__':
    import _context
from minitn.lib.numerical import PotentialFunction
from minitn.lib.symbolic import BasisFunction
from minitn.dvr import CasDVR


def test_dvr(x0, L, n, v_func):
    def f(x): return sym.cos(sym.pi * (x - x0) / L)

    def inv_f(y): return x0 + sym.acos(y) * L / sym.pi

    basis = [BasisFunction.particle_in_box(i, L, x0)
             for i in range(1, 1 + n)]
    dvr = CasDVR(
        basis, trans_func_pair=(f, inv_f), cut_off=(x0, x0 + L),
        num_prec=100
    )
    dvr.set_v_func(v_func)
    e, _ = dvr.solve(n_state=5)
    for i, e_i in enumerate(e):
        logging.info('e{}: {}'.format(i, e_i))
    dvr.plot_eigen(x0, x0 + L, npts=100)
    # dvr.plot_dvr(x0, x0 + L, npts=100)
    return


def test_improper_dvr(x0, L, n, v_func):
    # basis = [cas.harmonic_oscillator(i)
    #          for i in range(0, n)]
    basis = [BasisFunction.particle_in_box(i, L, x0)
             for i in range(1, 1 + n)]
    dvr = CasDVR(basis, cut_off=(x0, x0 + L), num_prec=100)
    dvr.comment += '-improper'
    dvr.set_v_func(v_func)
    e, _ = dvr.solve()
    for i, e_i in enumerate(e):
        logging.info('e{}: {}'.format(i, e_i))
    dvr.plot_eigen(x0, x0 + L, npts=100)
    dvr.plot_dvr(x0, x0 + L, npts=100)
    return


def main():
    import time
    x0, L, n = -5., 10., 10
    v_func = PotentialFunction().sho()
    t0 = time.time()
    test_dvr(x0, L, n, v_func)
    t1 = time.time()
    test_improper_dvr(x0, L, n, v_func)
    t2 = time.time()
    logging.info('Proper: {}; Improper: {}'.format(t1 - t0, t2 - t1))


if __name__ == '__main__':
    main()
