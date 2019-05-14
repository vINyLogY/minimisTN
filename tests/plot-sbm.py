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
        'sbm-ft-origin',
        'sbm-ft-split',
    ]
    with figure():
        tp_list = []
        for string in str_list:
            tmp = np.load(root + string + '.npy')
            t, p = zip(*tmp)
            plt.plot(t, p, '-', label=string.upper())
            tp_list.append(tmp)
        plt.legend(loc='best')
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.show()


if __name__ == '__main__':
    plot()
