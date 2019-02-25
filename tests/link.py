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
from minitn.dvr import SineDVR
from minitn.ml import Multi_layer

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

hlist = []    # \sum_i x_i^2 + \sum_i x_i * x_{i+1}
low, up, n_dvr = -5., 5., 1
dvr = SineDVR(low, up, n_dvr)
# single
square = lambda x: x ** 2
dvr.set_v_func(square)
s_h = dvr.h_mat()
for leaf in ham:
    hlist.append([(leaf, s_h)])
# couple
"""
linear = lambda x: x
dvr.set_v_func(linear)
l_h = dvr.v_mat()
for i in range(dofs - 1):
    term = [(ham[i], l_h), (ham[i + 1], l_h)]
    hlist.append(term)
"""

for i, term in enumerate(hlist):
    print('In term {}:'.format(i))
    for leaf, array in term:
        msg = "    leaf: {},\n    array: {}".format(leaf, array)
        print(msg)

solver = Multi_layer(mps[0], hlist)
diff = solver.eom()
for tensor, array in diff.items():
    msg = "tensor: {},\narray: {}".format(tensor, array)
    print(msg)


# EOF
