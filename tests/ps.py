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
from sho_model import test_4layers, test_2layers

def test_binary():
    exp = test_4layers(x0=-5., x1=5., n_1=5, n_2=5, n_3=5, n_4=40, c=0.5)
    root = exp.root

    root, _ = root.split(0, child=root)
    for t in root.visitor():
        t.check_completness(strict=True)
        print(t, np.sum(t.array), t.shape)
    for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
        print(i, j, k, l)
    logging.info(__('Norm:{:.8f}', root.global_norm()))

    root = root.unite(0, root=root)
    for t in root.visitor():
        t.check_completness(strict=True)
        print(t, np.sum(t.array), t.shape)
    for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
        print(i, j, k, l)
    logging.info(__('Norm:{:.8f}', root.global_norm()))

def test_others():
    exp = test_2layers(-5., 5., 40, 6, 4, 0.5)
    root = exp.root
    for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
        print(i, j, k, l)

    root, child = root.split(2, indice=(0, 2), child=root)
    for t in root.visitor(axis=None):
        t.check_completness(strict=True)
    for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
        print(i, j, k, l)
    logging.info(__('Norm:{:.8f}', root.global_norm()))

    root = child.unite(2, root=child)
    for t in root.visitor():
        t.check_completness(strict=True)
    for i, j, k, l in root.linkage_visitor(leaf=False, back=True):
        print(i, j, k, l)
    logging.info(__('Norm:{:.8f}', root.global_norm()))



logging.root.setLevel(logging.DEBUG+2)
# test_binary()
test_others()

