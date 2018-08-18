#!/usr/bin/env python2
# coding: utf-8
from __future__ import division, print_function

import logging
from builtins import zip

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, ifft

if __name__ == '__main__':
    import _context
from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import figure, BraceMessage as __


def main():
    x0, L, n = -5., 10., 40
    v_func = PotentialFunction.sho()
    vf_list = [v_func] * 2
    conf_list = [[x0, x0 + L, n]] * 2
    po_dvr = PO_DVR(conf_list, fast=False)
    # po_dvr.set_v_func(vf_list)
    # _, v = po_dvr.solve()
    c = 0.5
    logging.info(__('c: {:f}', c))
    v_rst = PotentialFunction.linear_corr(c)
    po_dvr.set_v_func(vf_list, v_rst=v_rst)
    length = 10.
    window = WindowFunction.g1prime(length)
    init = None
    # init = (po_dvr.mu_mat(0)).dot(po_dvr.init_state())
    # logging.debug(__(
    #     'norm_0: {}', scipy.linalg.norm(init)
    # ))
    # e0 = expection(po_dvr.h_mat(), po_dvr.init_state())
    freq, sigma = po_dvr.spectrum(
        init=init, length=length, max_inter=0.001, window=window)
    # freq -= e0
    # with figure() as fig1:
    #     plt.plot(t, np.abs(auto))
    #     plt.show()
    with figure() as fig2:
        plt.plot(freq, np.abs(sigma), '.')
        plt.plot(freq, np.abs(sigma), 'k-')
        plt.xlim(-1., 5.)
        plt.ylim(0., 1.)
        plt.show()

if __name__ == '__main__':
    main()
