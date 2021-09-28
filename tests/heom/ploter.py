#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from matplotlib import pyplot as plt

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'delta'))

prefix = "Tier 5"
a = np.loadtxt('HEOM_delta_t5_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], '-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '-', label="$|r|$ ({})".format(prefix))

prefix = "Tier 10"
a = np.loadtxt('HEOM_delta_t10_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], '-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '-', label="$|r|$ ({})".format(prefix))

prefix = "Tier 20"
a = np.loadtxt('HEOM_delta_t20_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], '-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '-', label="$|r|$ ({})".format(prefix))

prefix = "Tier 100"
a = np.loadtxt('HEOM_delta_t100_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, -1], '-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '-', label="$|r|$ ({})".format(prefix))

prefix = "Analytic"
g = 0.1
w = 0.05
t = a[:, 0]
r = np.exp(-32 * np.sqrt(2) * g**2 / w**2 * (1 - np.cos(w * t)))
plt.plot(t, r, 'k--', label="$|r|$ ({})".format(prefix))

plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.title('Delta model')
plt.savefig("tiers.png", bbox_inches='tight')
