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
    str2 = 'exp'
    str1 = 'ref'

    t1 = np.load(root + str1 + '_t.npy')
    a1 = np.load(root + str1 + '_a.npy')
    t2 = np.load(root + str2 + '_t.npy')
    a2 = np.load(root + str2 + '_a.npy')
    with figure():
        plt.plot(t2, np.abs(a2)**2, '-', label='MCTDH')
        plt.plot(t1, np.abs(a1)**2, '--', label='ML-MCTDH-PS')
        plt.legend(loc='best')
        plt.xlabel(r'$t$ (a. u.)')
        plt.ylabel(r'$P$')
        plt.show()



def plot_n():
    zipped = {}
    try:
        t_str = 'data/ed_t.npy'
        auto_str = 'data/ed_a.npy'
        t = np.load(t_str)
        auto = np.load(auto_str)
        zipped[0] = (t, auto)
    except:
        pass
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
            if i == 0:
                label = 'ED'
                plt.plot(t, np.abs(a) ** 2, 'k-', label=label)
            else:
                label = r"$N_1 = {0}$".format(i)
                plt.plot(t, np.abs(a) ** 2, '--', label=label)
        plt.legend(loc='best')
        plt.xlabel(r'$t$ (a. u.)')
        plt.ylabel(r'$P$')
        plt.show()


plot_n()
#plot()
