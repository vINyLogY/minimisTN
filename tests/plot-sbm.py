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
        # ('data/sbm-ft-split-origin', r'FT-PS-origin'),
        # ('tmp/sbm-ft-split', r'FT-PS'),
        ('sbm-zt-mctdh-origin', 'Std.'),
        ('data/sbm-zt-origin', 'MCTDH-0.25'),
        # ('tmp/sbm-zt-split-13000', 'MCTDH-PS-0.05'),
        # ('data/sbm-zt-split-0.25fs', 'MCTDH-PS-0.25'),
        ('./sbm-zt-split-snd-r', 'Test-snd (rec.)'),
        ('./sbm-zt-split-snd', 'Test-snd'),
        ('./sbm-zt-split-fst', 'Test-fst'),
        ('sbm-zt-split-fst-half', 'Test-fst (half)')
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
