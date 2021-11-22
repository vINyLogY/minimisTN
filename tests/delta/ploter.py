import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "1-DOF_TT_t10_heom.dat",
    "1-DOF_2site_t10_heom.dat",
]
heom_label_list = [
    "HEOM",
    "HEOM2",
]

wfn_list = [
    "1-DOF_simple_t10_wfn.dat",
    #    "1-DOF_2site_t10_wfn.dat",
]
wfn_label_list = [
    "WFN",
    #    "WFN2",
]

for fname, label in zip(wfn_list, wfn_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 1]), '-', label="$P_0$ ({})".format(label))
    #plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), '-', label="$|r|$ ({})".format(label))

for fname, label in zip(heom_list, heom_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 1]), '--', label="$P_0$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.real(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), '--', label="$|r|$ ({})".format(label))

plt.legend(loc='upper right')
title = 'SBM with relaxation (2 sites)'
plt.title(title)
os.chdir(f_dir)
plt.ylim(-1, 1)
plt.savefig('{}.png'.format(title))
