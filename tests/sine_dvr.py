#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging
import sys
from itertools import count

import matplotlib.pyplot as plt
import numpy as np

import _context
from minitn.dvr import FastSineDVR, SineDVR
from minitn.lib.numerical import PotentialFunction, expection
from minitn.lib.tools import __, figure


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
        logging.info('e{}: {}'.format(i, e_i))
    v0 = expection(sine_dvr.v_mat(), v[0])
    t0 = expection(sine_dvr.t_mat(), v[0])
    logging.info(__(
        'v0: {}; t0: {}', v0, t0
    ))
    sine_dvr.plot_eigen(npts=1000, n_plot=6, scale=1.)
    sine_dvr.plot_dvr(npts=1000)
    return


def test_percentage(x0, L, n_max, v_func, fast=False):
    e_list = []
    e1_list = []
    e2_list = []
    n_list = []
    for n in count(30, 1):
        if n > n_max:
            break
        if fast:
            sine_dvr = FastSineDVR(x0, x0 + L, n)
        else:
            sine_dvr = SineDVR(x0, x0 + L, n)
        sine_dvr.set_v_func(v_func)
        e, v = sine_dvr.solve()
        e_list.append(e[0])
        e1_list.append(e[1])
        e2_list.append(e[2])
        n_list.append(n)
    if fast:
        sine_dvr = FastSineDVR(x0, x0 + L, 400)
    else:
        sine_dvr = SineDVR(x0, x0 + L, 400)
    sine_dvr.set_v_func(v_func)
    e, v = sine_dvr.solve()
    e_list, e1_list, e2_list = map(
        np.array, (e_list, e1_list, e2_list))
    per0 = e_list / e[0]
    per1 = e1_list / e[1]
    per2 = e2_list / e[2]
    with figure():
        # plt.plot(n_list, per, 'kx')
        plt.plot(n_list, per0, 'r-')
        plt.plot(n_list, per1, 'g-')
        plt.plot(n_list, per2, 'b-')
        plt.plot(n_list, per0, 'rx')
        plt.plot(n_list, per1, 'gx')
        plt.plot(n_list, per2, 'bx')

        plt.savefig('percentage-{}.pdf'.format(
            'FastSineDVR' if fast else 'SineDVR'))
    return


def test_L_percentage(L_max, n, v_func, fast=False):
    e_list = []
    e1_list = []
    e2_list = []
    L_list = []
    for L in count(3.7, 0.1):
        if L > L_max:
            break
        x0 = L / 2.
        if fast:
            sine_dvr = FastSineDVR(-L/2, L/2, n)
        else:
            sine_dvr = SineDVR(-L/2, L/2, n)
        sine_dvr.set_v_func(v_func)
        e, v = sine_dvr.solve()
        e_list.append(e[0])
        e1_list.append(e[1])
        e2_list.append(e[2])
        L_list.append(L)
    if fast:
        sine_dvr = FastSineDVR(-L_max/2, L_max/2, 400)
    else:
        sine_dvr = SineDVR(-L_max/2, L_max/2, 400)
    sine_dvr.set_v_func(v_func)
    e, v = sine_dvr.solve()
    e_list, e1_list, e2_list = map(
        np.array, (e_list, e1_list, e2_list))
    per0 = e_list / e[0]
    per1 = e1_list / e[1]
    per2 = e2_list / e[2]
    with figure():
        # plt.plot(n_list, per, 'kx')
        plt.plot(L_list, per0, 'r-')
        plt.plot(L_list, per2, 'b-')
        plt.plot(L_list, per1, 'g-')

        plt.savefig('percentage-{}.pdf'.format(
            'FastSineDVR' if fast else 'SineDVR'))
    return



def main():
    import time
    x0, L, n = -5., 10., 400
    v_func = PotentialFunction.w_well()
    # t0 = time.time()
    # test_sine_dvr(x0, L, n, v_func, n_plot=5, fast=True)
    # t1 = time.time()
    # test_sine_dvr(x0, L, n, v_func, n_plot=5, fast=False)
    # t2 = time.time()
    # logging.info('fast: {}, dense: {}'.format(t1 - t0, t2 - t1))
    # test_percentage(x0, L, 60, v_func, fast=False)
    test_L_percentage(L, n, v_func, fast=False)


if __name__ == '__main__':
    main()
