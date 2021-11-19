import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "re_fine-1-T30_heom.dat",
]
heom_label_list = [
    "HEOM",
]

wfn_list = [
    "re_fine-1-T30_wfn.dat",
]
wfn_label_list = [
    "WFN",
]

for fname, label in zip(wfn_list, wfn_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(tst[:, 0], np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(label))
    #plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    #plt.plot(tst[:, 0], np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(label))
    #plt.plot(tst[:, 0], np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(label))
    plt.plot(tst[:, 0], np.abs(tst[:, 2]), '-', label="$|r|$ ({})".format(label))

for fname, label in zip(heom_list, heom_label_list):
    tst = np.loadtxt(fname, dtype=complex)

    plt.plot(tst[:, 0], np.abs(tst[:, 1]), '--', label="$P_0$ ({})".format(label))
    #plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(label))
    #plt.plot(tst[:, 0], np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(label))
    #plt.plot(tst[:, 0], np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(label))
    plt.plot(tst[:, 0], np.abs(tst[:, 2]), '-.', mfc='none', label="$|r|$ ({})".format(label))

plt.legend(loc='lower right')
title = 'SBM with relaxation'
plt.title(title)
os.chdir(f_dir)
plt.savefig('{} (t30 long).png'.format(title))
