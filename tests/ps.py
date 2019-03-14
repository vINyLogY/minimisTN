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
print(repr(root))
root, child = root.split(0, root=root)
print(repr(root))
root.check_completness(strict=True)
for t in root.visitor():
    print(t, np.sum(t.array), t.shape)
print('*' * 10)
for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
    print(i, j, k, l)
logging.info(__('Norm:{:.8f}', root.global_norm()))
print('*' * 10)
print(repr(root))
root = root.unite(0, root=root)
print(repr(root))
root.check_completness(strict=True)
for t in root.visitor():
    print(t, np.sum(t.array), t.shape)
print('*' * 10)
for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
    print(i, j, k, l)
logging.info(__('Norm:{:.8f}', root.global_norm()))
