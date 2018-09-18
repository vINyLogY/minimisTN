#!/usr/bin/env python2
# coding: utf-8
from __future__ import absolute_import, division

import logging
import sys
from builtins import input
from math import sqrt
from functools import partial

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import _context
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import time_this, figure
from minitn.mctdh import MCTDH


def linear(x, c=0.1):
    return sqrt(c) * x


@time_this
def test_mctdh(x0, L, m, n, v_func, c):
    vf_list = [v_func] * 2
    conf_list = [[x0, x0 + L, n]] * 2
    shape_list = [(n, m)] * 2
    case = MCTDH(conf_list, shape_list)
    case.set_v_func(vf_list)
    lin = partial(linear, c=c)
    # ex = []    # H_rst = 0
    ex = [[(0, linear), (1, linear)]]    # H_rst = 0.5xy
    case.gen_h_terms(extra=ex, kinetic_only=False)
    init = case.init_state()
    logging.info(__('shape of init vec: {}', init.shape))
    logging.info(__(
        'E0: {:.8f}', case.expection(init)
    ))
    logging.info('=' * 60)
    length = 30.
    window = WindowFunction.g0prime(length)
    t, auto = zip(*case.autocorrelation(
        stop=length, max_inter=0.001, const_energy=True,
        renormalize=True
        ))
    # freq, sigma = case.spectrum(
    #     length=length, max_inter=0.001, window=None
    # )
    with figure() as fig:
        plt.plot(t, np.abs(auto), '.')
        plt.plot(t, np.abs(auto), 'k-')
        namestr = 'MCTDH-C{}.svg'.format(c)
        plt.savefig(namestr)
    return


def main():
    import time
    x0, L, m, n = -5., 10., 10, 40
    v_func = PotentialFunction.sho()
    for i in range(30):
        c = i * 0.01
        test_mctdh(x0, L, m, n, v_func, c=c)


if __name__ == '__main__':
    main()
