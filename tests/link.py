#!/usr/bin/env python
# coding: utf-8
r"""Link test
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.lib.tools import __
from minitn.tensor import Tensor, Leaf

logging.root.setLevel(logging.DEBUG)


mps = []
spf = []
ham = []
dofs = 5
for i in range(dofs):
    temp = Tensor(axis=0, name='M' + str(i))
    array = [[1]] if i in [0, dofs - 1] else [[[1]]]
    temp.set_array(array)
    mps.append(temp)
for i in range(dofs):
    temp = Tensor(axis=0, name='S' + str(i))
    temp.set_array([[1]])
    spf.append(temp)
for i in range(dofs):
    temp = Leaf(name='H' + str(i))
    temp.reset()
    ham.append(temp)

# left-normalized
mps[0].axis = None
mps[0].link_to(0, spf[0], 0)
mps[0].link_to(1, mps[1], 0)
for i in range(1, dofs - 1):
    mps[i].link_to(1, spf[i], 0)
    mps[i].link_to(2, mps[i + 1], 0)
mps[-1].link_to(1, spf[-1], 0)
for i in range(dofs):
    spf[i].link_to(1, ham[i], 0)

for i in mps[0].visitor():
    print(i)

print('*'*10)

for a, i, b, j in mps[0].linkage_visitor():
    print("({0}, {1}) to ({2}, {3})".format(a, i, b, j))

m0, m1 = mps[0], mps[1]
s0 = spf[0]

for tensor in m0.visitor():
    tensor.check_completness(strict=True)

print(m1.projector())
print(m1.partial_env(1))
print(m1.partial_env(0))




# EOF
