#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import numpy as np

data = np.loadtxt("sbm-eq-split-snd", dtype=complex)
t = data[:, 0]
p = data[:, 1]
pr = data[:, 2]
plt.plot(t, np.abs(p), '-')
plt.plot(t, np.abs(pr),  '-')
plt.show()




