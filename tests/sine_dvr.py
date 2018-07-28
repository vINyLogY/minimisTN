#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging
import sys

import _context
from minitn.mydvr import FastSineDVR, SineDVR
from minitn.mycas import PotentialFunction


def test_sine_dvr(x0, L, n, v_func, n_plot=None, message=None, fast=False):
    if fast:
        sine_dvr = FastSineDVR(x0, x0 + L, n)
    else:
        sine_dvr = SineDVR(x0, x0 + L, n)
    if message is not None:
        sine_dvr.method = message
    sine_dvr.set_v_func(v_func)
    e, v = sine_dvr.solve()
    for i, e_i in enumerate(e[:6]):
        print('e{}: {}'.format(i, e_i))
    # sine_dvr.plot_eigen(npts=100, n_plot=n_plot, scale=1.)
    # sine_dvr.plot_dvr(npts=100)
    return


def main():
    import time
    x0, L, n = -5., 10., 1000
    v_func = PotentialFunction().sho()
    t0 = time.time()
    test_sine_dvr(x0, L, n, v_func, n_plot=5, message='SHO', fast=True)
    t1 = time.time()
    test_sine_dvr(x0, L, n, v_func, n_plot=5, message='SHO', fast=False)
    t2 = time.time()
    print('fast: {}, dense: {}'.format(t1 - t0, t2 - t1))


if __name__ == '__main__':
    main()
