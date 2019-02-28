#!/usr/bin/env python2
# coding: utf-8
from __future__ import division, print_function

import logging
from builtins import zip

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.fftpack import fft, ifft

# if __name__ == '__main__':
#     import _context
from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import figure, BraceMessage as __


def main():
    try:
        t = np.load('t_auto.npy')
        auto = np.load('auto.npy')
    except:
        x0, L, n = -5., 10., 40
        v_func = PotentialFunction.sho()
        vf_list = [v_func] * 2
        conf_list = [[x0, x0 + L, n]] * 2
        po_dvr = PO_DVR(conf_list, fast=False)
        # po_dvr.set_v_func(vf_list)
        # _, v = po_dvr.solve()
        c = 0.25
        logging.info(__('c: {:f}', c))
        v_rst = PotentialFunction.linear_corr(c)
        po_dvr.set_v_func(vf_list, v_rst=v_rst)
        length = 1.
        # init = (po_dvr.mu_mat(0)).dot(po_dvr.init_state())
        # logging.debug(__(
        #     'norm_0: {}', scipy.linalg.norm(init)
        # ))
        # e0 = expection(po_dvr.h_mat(), po_dvr.init_state())
        t, auto = zip(*po_dvr.autocorrelation(
            stop=length, max_inter=0.001, const_energy=True
            ))
        np.save('t_auto', t)
        np.save('auto', auto)

    # Plot
    zipped = []
    for m in range(1, 11):
        if m % 2:
            # continue
            pass
        try:
            t_m = np.load('t_auto_MCTDH_{}.npy'.format(m))
            auto_m = np.load('auto_MCTDH_{}.npy'.format(m))
            zipped.append((m, t_m, auto_m))
        except:
            pass
    with figure():
        l, = plt.plot(
            t, np.abs(auto), '-',
            label='\\textrm{Standard procedure}'
        )
        handles = [l]
        for m, t_m, auto_m in zipped:
            l, = plt.plot(
                t_m, np.abs(auto_m), '--',
                label=r'''$n^{(1)}'''+' = {}$'.format(m)
            )
            handles.append(l)
        plt.xlim(xmin=0, xmax=1)
        plt.xlabel(r"""$t$""")
        plt.ylabel(r"""$|a(t)|$""")
        plt.legend(handles=handles, loc='best')
        plt.savefig('autocorr_MCTDH.pdf')


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    main()
