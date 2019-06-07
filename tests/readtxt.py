#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

import logging
import os
import sys
from builtins import filter, map, range, zip
import re
import numpy as np

from minitn.lib.tools import plt, figure, BraceMessage as __
from minitn.lib.units import Quantity

filename = 'ft-so.txt'
with open(filename, 'r') as f:
    fs = f.read()
    time = []
    for m in re.finditer(r"Time: (.*) fs", fs):
        li = m.group().split(' ')
        time.append(float(li[1]))
    cmf_steps = []
    for m in re.finditer(r"steps: (.*)", fs):
        li = m.group().split(' ')
        cmf_steps.append(int(li[1])+1)
        if cmf_steps[-1] == 0:
            pass
    e = []
    for m in re.finditer(r"E: (.*?)[\+|\-]", fs):
        li = m.group().split(' ')
        e.append(Quantity(float(li[1][:-1])).convert_to('cm-1').value)
    de = [ei - e[0] for ei in e]
    with figure():
        plt.plot(np.array(time[:-1]), np.array(de), '-')
        plt.xlim(0, 100)
        plt.xlabel(r'$t$ (fs)')
        plt.ylabel(r'$\Delta U$ ($\mathrm{cm}^{-1}$)')
        plt.show()
        print(list(zip(time[:-1], cmf_steps)))


