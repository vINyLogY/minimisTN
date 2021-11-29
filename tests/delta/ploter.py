import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "boson_300K_t5_heom.dat",
    "drude_300K_t5_heom.dat",
]
heom_label_list = [
    "Boson",
    "Drude",
]

wfn_list = [
    "drude_boson_300K_t5_heom.dat",
]
wfn_label_list = [
    "Drude + Boson",
]

for fname, label in zip(wfn_list, wfn_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 1]), 'k-', label="$P_0$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '-', label="$\Re r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '-', label="$\Im r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), 'b-', label="$|r|$ ({})".format(label))

for fname, label in zip(heom_list, heom_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 1]), '--', label="$P_0$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(label))
    #plt.plot(np.real(tst[:, 0]), np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(label))
    plt.plot(np.real(tst[:, 0]), np.abs(tst[:, 2]), '--', label="$|r|$ ({})".format(label))

plt.legend(loc='upper left')
title = 'SBM with relaxation'
plt.title(title)
os.chdir(f_dir)
plt.ylim(0, 1)
plt.xlim(0, 20)
plt.savefig('{}.png'.format(title))
