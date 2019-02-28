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

@time_this
def main():
    x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 2
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    ref = test_mctdh(x0, x1, n_dvr, n_spf, c)
    # e0 = exp.root.vectorize()
    # r0 = np.concatenate((ref.vec[800:], ref.vec[:800]), axis=None)
    for i, _ in enumerate(
        zip(exp.propagator(method='RK45'), ref.propagation())
    ):
        vec1 = exp.root.vectorize()
        v = ref.vec
        f1, f2 = v[:800], v[800:]
        vec2 = np.concatenate([f2, f1], axis=None)
        print('sum: {}'.format(np.sum(np.abs(vec1 - vec2))))
        if i > 10000:
            break

    return


logging.root.setLevel(logging.INFO)
main()
