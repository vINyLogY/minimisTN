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


t1 = np.load('ml_t.npy')
a1 = np.load('ml_a.npy')
t2 = np.load('mctdh_t.npy')
a2 = np.load('mctdh_a.npy')
with figure():
    plt.plot(t2, np.abs(a2), '-')
    plt.plot(t1, np.abs(a1), '--')
    plt.show()
