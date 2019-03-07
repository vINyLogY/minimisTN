#!/usr/bin/env python
# coding: utf-8
r"""Test Tensor.generate()
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip

from minitn.tensor import Tensor

def main():
    root = 0
    graph = {
        0: [1, 2, 3],
        1: [4, 5],
        2: [6],
        3: [7, 8, 9]
    }

    a = Tensor.generate(graph, root)
    for i, j, k, l in a.linkage_visitor():
        print(i, j, k, l)

logging.root.setLevel(logging.DEBUG)
main()
