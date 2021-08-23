#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from matplotlib import pyplot as plt

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'pyheom'))

prefix = "HEOM"
a = np.loadtxt('brownian.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], 'b-', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], 'r-', label="$P_1$ ({})".format(prefix))
#plt.plot(a[:, 0], np.real(a[:, 2]), '-', label="$\Re r$ ({})".format(prefix))
#plt.plot(a[:, 0], np.imag(a[:, 2]), '-', label="$\Im r$ ({})".format(prefix))

prefix = "Ref"
a = np.loadtxt('brownian_ref.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], 'k-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], 'k-.', label="$P_1$ ({})".format(prefix))
#plt.plot(a[:, 0], np.real(a[:, 2]), '--', label="$\Re(r)$ ({})".format(prefix))
#plt.plot(a[:, 0], np.imag(a[:, 2]), '--', label="$\Im(r)$ ({})".format(prefix))

plt.legend()
plt.savefig("brownian.png")
