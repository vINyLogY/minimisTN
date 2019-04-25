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
    str2 = 'ref'
    str1 = 'exp'

    t1 = np.load(root + str1 + '_t.npy')
    a1 = np.load(root + str1 + '_a.npy')
    t2 = np.load(root + str2 + '_t.npy')
    a2 = np.load(root + str2 + '_a.npy')
    t3 = np.load(root + str1 + '2_t.npy')
    a3 = np.load(root + str1 + '2_a.npy')
    with figure():
        plt.plot(t2, np.abs(a2), '-', label='MCTDH')
        plt.plot(t1, np.abs(a1), '--', label='ML-MCTDH-PS-U')
        plt.plot(t3, np.abs(a3), '--', label='ML-MCTDH-PS-S')
        plt.legend(loc='best')
        plt.show()



def plot_n():
    zipped = {}
    for i in range(5):
        t_str = 'data/ref_{}_t.npy'.format(i)
        auto_str = 'data/ref_{}_a.npy'.format(i)
        try:
            t = np.load(t_str)
            auto = np.load(auto_str)
            zipped[i] = (t, auto)
        except:
            pass
    with figure():
        for i, (t, a) in zipped.items():
            label = r'$N_1 = {0}$'.format(i)
            plt.plot(t, np.abs(a), '--', label=label)
        plt.legend(loc='best')
        plt.show()


#plot_n()
plot()
