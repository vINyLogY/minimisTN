import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "2-DOF_MPS_T10_en_heom.dat",
]
heom_label_list = [
    "HEOM",
]

wfn_list = [
    "2-DOF_MPS_T10_en_wfn.dat",
]
wfn_label_list = [
    "WFN",
]

for fname, label in zip(wfn_list, wfn_label_list):
    tst = np.loadtxt(fname, dtype=complex)
    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 1]), '-', label="$E_S$ ({})".format(label))

for fname, label in zip(heom_list, heom_label_list):
    tst = np.loadtxt(fname, dtype=complex)
    plt.plot(np.real(tst[:, 0]), np.real(tst[:, 1]), '--', label="$E_S$ ({})".format(label))

plt.legend(loc='lower right')
title = 'SBM Energy (system)'
plt.title(title)
os.chdir(f_dir)
plt.savefig('{}.png'.format(title))
