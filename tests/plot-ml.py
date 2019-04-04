#!/usr/bin/env python2
# # coding: utf-8
from __future__ import division, print_function

import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

from minitn.lib.tools import figure, BraceMessage as __

root = 'data/'
str2 = 'ref'
str1 = 'exp'

t1 = np.load(root + str1 + '_t.npy')
a1 = np.load(root + str1 + '_a.npy')
t2 = np.load(root + str2 + '_t.npy')
a2 = np.load(root + str2 + '_a.npy')
for n, (i1, i2) in enumerate(zip(a1, a2)):
    d = (i1 - i2)
    if abs(d) > 1.e-10:
        print(n, i1, i2)
with figure():
    plt.plot(t2, np.real(a2), '-')
    plt.plot(t1, np.real(a1), '--')
    plt.show()
with figure():
    plt.plot(t2, np.imag(a2), '-')
    plt.plot(t1, np.imag(a1), '--')
    plt.show()
with figure():
    plt.plot(t2, np.abs(a2), '-')
    plt.plot(t1, np.abs(a1), '--')
    plt.show()
