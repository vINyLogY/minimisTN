#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging
import os
import sys

import numpy as np

from minitn.dvr import SineDVR
from minitn.lib.numerical import PotentialFunction


def test_propagation(x0, L, n):
    sine_dvr = SineDVR(x0, x0 + L, n)
    v_func = PotentialFunction().sho(k=3., x0=-1.)
    sine_dvr.set_v_func(v_func)
    _, v_list = sine_dvr.solve(n_state=1)
    v = v_list[0]
    # func = sine_dvr.dvr2cont(v)
    # func_list = [func]
    # ifunc_list = []
    v_func = PotentialFunction().w_well()
    sine_dvr.set_v_func(v_func)
    sine_dvr.solve(n_state=1)
    p1, _, p3 = sine_dvr.propagator(tau=0.01, method='Trotter')
    logging.info(sine_dvr.energy_expection(v))
    for _ in range(1, 10 + 1):
        v = np.dot(p1, np.dot(p3, np.dot(p1, v)))
        logging.info(sine_dvr.energy_expection(v))
        # if i % 1 == 0:
        #     func = sine_dvr.dvr2cont(np.real(v))
        #     ifunc = sine_dvr.dvr2cont(np.imag(v))
        #     func_list.append(func)
        #     ifunc_list.append(ifunc)
        #     sine_dvr.method = 'fig-{:.2f}'.format(i * 0.01)
        #     sine_dvr.plot_func(
        #       [func, ifunc], y_min=-1.5, y_max=1.5, npts=200)
    return


def main():
    x0, L, n = -10., 20., 1000
    test_propagation(x0, L, n)


if __name__ == '__main__':
    main()
