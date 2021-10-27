#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from matplotlib import pyplot as plt

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'drude_yan2021'))

prefix = 'simple'
a = np.loadtxt('HEOM_{}_d1.0_tst.dat'.format(prefix), dtype=complex)
#a = np.loadtxt('HEOM_{}_k3_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '-.', label="$|r|$ ({})".format(prefix))

prefix = "TT"
a = np.loadtxt('HEOM_{}_d1.0_tst.dat'.format(prefix), dtype=complex)
#a = np.loadtxt('HEOM_{}_k3_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '--', label="$P_0$ ({})".format(prefix))
plt.plot(a[:, 0], 2 * np.abs(a[:, 2]), '--', label="$|r|$ ({})".format(prefix))

plt.legend(
    loc='center left',
    bbox_to_anchor=(1, 0.5),
)
plt.title('Drude model (w/ Yan2021)')
plt.savefig("vs_yan2021.png", bbox_inches='tight')
