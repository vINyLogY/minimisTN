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
    str1 = 'sbm-ft'

    tp1 = np.load(root + str1 + '.npy')
    t1, p1 = zip(*tp1)
    with figure():
        plt.plot(t1, p1, '-', label='ML-MCTDH-PS')
        plt.legend(loc='best')
        plt.show()


if __name__ == '__main__':
    plot()
