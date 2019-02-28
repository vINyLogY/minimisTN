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


t = np.load('t_auto_ML_100.npy')
auto = np.load('auto_ML_100.npy')
with figure():
    plt.plot(t, np.abs(auto), '.')
    plt.plot(t, np.abs(auto), 'k-')
    plt.show()
