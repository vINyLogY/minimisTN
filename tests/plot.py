#!/usr/bin/env python2
# # coding: utf-8
from __future__ import division, print_function

import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

if __name__ == '__main__':
    import _context
from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import figure, BraceMessage as __


for i in range(30):
    c = i*0.01
    t_str = 'data/t{}.npy'.format(c)
    auto_str = 'data/auto{}.npy'.format(c)
    t = np.load(t_str)
    auto = np.load(auto_str)
    with figure():
        plt.plot(t, np.abs(auto), '.')
        plt.plot(t, np.abs(auto), 'k-')
        namestr = 'MCTDH-C{}.svg'.format(c)
        plt.savefig(namestr)
