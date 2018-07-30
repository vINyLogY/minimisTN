#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging
import sys

import numpy as np

import _context
from minitn.mydvr import PO_DVR
from minitn.mycas import PotentialFunction


def test_po_dvr(x0, L, n, v_func, c=0.0, fast=True):
    print('Fast = {}'.format(fast))
    vf_list = [v_func] * 2
    conf_list = [[x0, x0 + L, n]] * 2
    po_dvr = PO_DVR(conf_list, fast=fast)
    for i in range(10):
        c = i * 0.01
        v_rst = PotentialFunction().linear_corr(i * 0.01)
        po_dvr.set_v_func(vf_list, v_rst=v_rst)
        e, _ = po_dvr.solve(n_state=6)
        print('c: {:.2f}; e: {}'.format(i * 0.01, e))
    return


def ref(n_state):
    print('Reference')
    for i in range(10):
        c = i * 0.01
        e_1 = np.sqrt(1 - c)
        e_2 = np.sqrt(1 + c)
        l1 = [(n + 0.5) * e_1 for n in range(100)]
        l2 = [(n + 0.5) * e_2 for n in range(100)]
        l_ = []
        for a in l1:
            for b in l2:
                l_.append(a + b)
        e = np.array(sorted(l_))[:n_state]
        print('c: {:.2f}; e: {}'.format(i * 0.01, e))
    return


def main():
    import time
    x0, L, n = -5., 10., 40
    v_func = PotentialFunction().sho()
    t0 = time.time()
    test_po_dvr(x0, L, n, v_func, fast=True)
    t1 = time.time()
    test_po_dvr(x0, L, n, v_func, fast=False)
    t2 = time.time()
    print('fast: {}, dense: {}'.format(t1 - t0, t2 - t1))
    ref(6)


if __name__ == '__main__':
    main()
