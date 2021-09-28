#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from matplotlib import pyplot as plt

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(f_dir)

prefix = "Train"
a = np.loadtxt('HEOM_train_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], 'b-', label=r"$Tr \rho_0$ ({})".format(prefix))
#plt.plot(a[:, 0], np.real(a[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
#plt.plot(a[:, 0], np.imag(a[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

plt.legend()
plt.savefig("drude_trace.png")
