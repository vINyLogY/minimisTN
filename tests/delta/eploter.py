import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'data'))
heom_list = [
    "boson_scale_300K_t5_heom.dat",
]
heom_label_list = [
    "Boson 5",
]

wfn_list = [
    "boson_scale_300K_t15_heom.dat",
    #"drude_boson_300K_t5_heom.dat",
]
wfn_label_list = [
    "Boson 15",
    #"Drude + Boson",
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
