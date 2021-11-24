import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "drude_boson_simple_t10_heom.dat",
    "drude_simple_t10_heom.dat",
    "boson_simple_t10_heom.dat",
    "boson_simple_zt_t10_heom.dat",
]
heom_label_list = [
    "Drude + Boson",
    "Drude",
    "Boson",
    "Boson (ZT)",
]

wfn_list = [
    "boson_simple_zt_t10_wfn.dat",
]
wfn_label_list = [
    "Boson (ZT)",
]

for fname, label in zip(wfn_list, wfn_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), '-', label="$|r|$ ({})".format(label))

for fname, label in zip(heom_list, heom_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), '--', label="$|r|$ ({})".format(label))

plt.legend(loc='lower right')
title = 'SBM with relaxation'
plt.title(title)
os.chdir(f_dir)
plt.ylim(-1, 1)
plt.savefig('{}.png'.format(title))
