#!/usr/bin/env python
# coding: utf-8
from matplotlib import rc
rc('font', family='Times New Roman')
rc('text', usetex=True)

from matplotlib import pyplot as plt
import numpy as np


data_list = [
    # "1_150_const-coupling_time-pop-purity_split-snd.txt",
    # "1_400_const-coupling_time-pop-purity_split-snd.txt",
    # "1_650_const-coupling_time-pop-purity_split-snd.txt",
    # "1_2100_const-coupling_time-pop-purity_split-snd.txt",
    "1_1979_const-coupling_time-pop-purity_split-snd.txt",
    "4_const-coupling_time-pop-purity_split-snd.txt",
    '1_formula.txt',
    # '1_unrot_formula.txt',
    # 'test_750_split-snd.txt',
]
for data_name in data_list:
    data = np.loadtxt(data_name, dtype=complex)
    label = ';'.join(data_name.split('_')[:2])
    t = data[:, 0]
    p = data[:, 1]
    pr = data[:, 2]
    pattern = '-' if label.startswith('4') else '-'
    #plt.plot(t, np.abs(p), pattern, label='{}: Population'.format(label))    
    plt.plot(t, np.abs(pr), pattern, label='L_2^2')
    plt.xlabel('Time')
    plt.ylabel('Entropy')
#plt.legend()
plt.show()




