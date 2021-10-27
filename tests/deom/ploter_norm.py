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
a = np.loadtxt('HEOM_2site_l2.5_w0.2_tst.dat_norm'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], '-.', label="Norm ({})".format(prefix))

prefix = "$\lambda$ 0.25, $\omega_c$ 2.5"
a = np.loadtxt('HEOM_2site_l0.2_w2.5_tst.dat_norm'.format(prefix), dtype=complex)
plt.plot(a[:, 0], a[:, 1], 'o', markerfacecolor='none', label="Norm ({})".format(prefix))

plt.legend(loc='best')
plt.title('Drude model')
plt.xlim(0, 20)
plt.savefig("diff_norm.png", bbox_inches='tight')
