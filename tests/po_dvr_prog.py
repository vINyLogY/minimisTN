#!/usr/bin/env python2
# coding: utf-8
from __future__ import division

import logging

import numpy as np

if __name__ == '__main__':
    import _context
from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction
from minitn.lib.tools import BraceMessage as __


def main():
    logging.root.setLevel(logging.INFO)
    x0, L, n = -5., 10., 40
    v_func = PotentialFunction.sho()
    vf_list = [v_func] * 2
    conf_list = [[x0, x0 + L, n]] * 2
    po_dvr = PO_DVR(conf_list, fast=False)
    c = 0.5
    logging.info(__('c: {:f}', c))
    v_rst = PotentialFunction.linear_corr(c)
    po_dvr.set_v_func(vf_list, v_rst=v_rst)
    _, v = po_dvr.solve()
    po_dvr.plot_propagation(stop=100., max_inter=0.01)
    # for i in po_dvr.propagation(stop=1., max_inter=0.01):
    #     continue

if __name__ == '__main__':
    main()
