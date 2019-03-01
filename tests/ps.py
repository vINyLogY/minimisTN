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
from sho_model import test_2layers


logging.root.setLevel(logging.DEBUG)
x0, x1, n_dvr, n_spf, c, dofs = -5., 5., 40, 10, 0.5, 2
exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
root = exp.root
root = root.split(0)
root.check_completness(strict=True)
for t in root.visitor():
    print(t, np.sum(t.array))
logging.info(__('before Norm:{:.8f}', root.global_norm()))
print('*' * 10)
root = root.unite(1)
root.check_completness(strict=True)
for t in root.visitor():
    print(t, np.sum(t.array))
logging.info(__('after Norm:{:.8f}', root.global_norm()))


