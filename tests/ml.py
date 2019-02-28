#!/usr/bin/env python
# coding: utf-8
r"""ML type of 2d sho
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.lib.tools import __, time_this, figure
from tests.sho_model import test_2layers


@time_this
def main():
    solver = test_2layers(
        lower=-5., upper=5., n_dvr=40, n_spf=5, dofs=2, c=0.5
    )
    h = 0.001
    cmf_step = None
    i = 0
    for _, r in solver.propagator(
        end=100, ode_inter=h, cmf_step=cmf_step, method='RK45'
    ):
        if cmf_step is None or i == 0 or i % cmf_step == 0:
            print('* t: {}'.format(i * h))
            for tensor in r.visitor(leaf=False):
                norm = np.sum(tensor.local_norm())
                msg = "tensor: {}, norm: {}".format(
                    tensor, norm
                )
                print(msg)
            logging.info(__(
                'energy: {}', r.expection()
            ))
        i += 1
    return


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG + 1)
    main()

# EOF
