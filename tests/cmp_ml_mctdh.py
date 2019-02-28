#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.lib.tools import __, time_this
from sho_model import test_2layers, test_mctdh

def main():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 2
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    ref = test_mctdh(x0, x1, n_dvr, n_spf, c)
    e0 = exp.root.vectorize()
    r0 = np.concatenate((ref.vec[800:], ref.vec[:800]), axis=None)
    print('t0:')
    for i, n in enumerate(e0 - r0):
        if abs(n) > 1.e-14:
            print('i: {}, n: {}'.format(i, n))
    exp.eom()
    exp.eom()    # test side effects
    vec1 = exp.root.vectorize(use_aux=True)
    ref_op = ref.h_mat()
    v = ref_op(ref.vec) / 1.0j
    f1, f2 = v[:800], v[800:]
    vec2 = np.concatenate([f2, f1], axis=None)
    print('t1:')
    for i, n in enumerate(e0 - r0  + 0.1 * vec1 - 0.1 * vec2):
        if abs(n) > 1.e-14:
            print('i: {}, n: {}'.format(i, n))

    return


logging.root.setLevel(logging.DEBUG+1)
main()
