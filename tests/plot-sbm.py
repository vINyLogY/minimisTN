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
    root = 'data/'
    str_list = [
        # 'sbm-ft-split-origin-1',
        # 'sbm-ft-split-origin-2',
        ('sbm-zt-2-split', 'ML-MCTDH-PS'),
        ('sbm-zt-origin', 'ML-MCTDH')
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
            plt.plot(t, p, '-', label=label)
            tp_list.append(tmp)
        plt.xlim(0, 100)
        #plt.legend(loc='best')
        plt.xlabel(r'$t$ (fs)')
        plt.ylabel(r'$P_2$')
        plt.show()


if __name__ == '__main__':
    plot()
