#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function
from minitn.heom.network import simple_heom

import os
from builtins import filter, map, range, zip

import numpy as np
from matplotlib import pyplot as plt

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))

prefix = "$\lambda$ 2.5, $\omega_c$ 0.25"
a = np.loadtxt('HEOM_2site_l2.5_w0.2_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1] / (a[:, 1] + a[:, -1]), 'k-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, 2] / (a[:, 1] + a[:, -1]), '-.', label="$r$ ({})".format(prefix))
a = np.loadtxt('HEOM_2site_l2.5_w0.2_ref.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1] / (a[:, 1] + a[:, -1]), 'k-', label="$P_1$ ({})".format(prefix))

prefix = "$\lambda$ 0.25, $\omega_c$ 2.5"
a = np.loadtxt('HEOM_2site_l0.2_w2.5_tst.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1] / (a[:, 1] + a[:, -1]), 'r-.', label="$P_1$ ({})".format(prefix))
plt.plot(a[:, 0], a[:, 2] / (a[:, 1] + a[:, -1]), '-.', label="$r$ ({})".format(prefix))
a = np.loadtxt('HEOM_2site_l0.2_w2.5_ref.dat'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1] / (a[:, 1] + a[:, -1]), 'r-', label="$P_1$ ({})".format(prefix))

plt.legend(loc='best',)
plt.title('Drude model')
plt.xlim(0, 20)
plt.savefig("diff_model.png", bbox_inches='tight')
