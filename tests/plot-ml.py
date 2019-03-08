#!/usr/bin/env python2
# # coding: utf-8
from __future__ import division, print_function

import logging
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy

from minitn.dvr import PO_DVR
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import figure, BraceMessage as __


str2 = 'ml'
str1 = '3l'

t1 = np.load(str1 + '_t.npy')
a1 = np.load(str1 + '_a.npy')
t2 = np.load(str2 + '_t.npy')
a2 = np.load(str2 + '_a.npy')
with figure():
    plt.plot(t2, np.abs(a2), '-')
    plt.plot(t1, np.abs(a1), '--')
    plt.show()
