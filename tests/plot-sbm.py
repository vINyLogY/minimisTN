#!/usr/bin/env python2
# coding: utf-8
from __future__ import division, print_function

import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

from minitn.lib.tools import figure, BraceMessage as __

def plot():
    root = './'
    str_list = [
        ('data/sbm-ft-split-origin', 'FT'),
        ('data/sbm-ft-split-origin-2', 'FT2'),
        ('sbm-zt-mctdh-origin', 'ZT'),
        ('tmp/sbm-zt-origin-13000', 'MCTDH'),
        ('tmp/sbm-zt-split-13000', 'MCTDH-PS'),
        ('tmp/sbm-ft-split', 'FT3')
    ]
    with figure():
        tp_list = []
        for string in str_list:
            try:
                name, label = string
            except:
                name = string
                label = None
            tmp = np.load(root + name + '.npy')
            t, p = zip(*tmp)
            if label == 'Std.':
                plt.plot(t, p, '-', label=label)
            else:
                plt.plot(t, p, '--', label=label)
            tp_list.append(tmp)
        plt.xlim(0, 100)
        plt.legend(loc='best')
        plt.xlabel(r'$t$ (fs)')
        plt.ylabel(r'$P_2$')
        plt.show()


if __name__ == '__main__':
    plot()
