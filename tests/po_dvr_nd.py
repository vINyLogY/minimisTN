#!/usr/bin/env python2
# coding: utf-8
from __future__ import division, absolute_import

import logging

import numpy as np

if __name__ == '__main__':
    import _context
from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction, expection
from minitn.lib.tools import time_this


@time_this
def test_po_dvr(x0, L, n, v_func, dim, fast=True, davidson=False):
    logging.debug('Fast = {}'.format(fast))
    vf_list = [v_func] * dim
    conf_list = [[x0, x0 + L, n]] * dim
    po_dvr = PO_DVR(conf_list, fast=fast)
    po_dvr.set_v_func(vf_list)
    e, v = po_dvr.solve(n_state=6, davidson=davidson)
    logging.info('e: {}'.format(e))
    e0 = expection(po_dvr.h_mat(), v[0])
    logging.info('e0: {:.8f}'.format(e0))
    return


def main():
    logging.getLogger().setLevel(logging.INFO)
    x0, L, n = -5., 10., 10
    v_func = PotentialFunction.sho()
    test_po_dvr(x0, L, n, v_func, 4, fast=False, davidson=False)


if __name__ == '__main__':
    main()
