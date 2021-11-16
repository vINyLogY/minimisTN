import os
from matplotlib import pyplot as plt
import numpy as np
import sys

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(f_dir)
prefix = "dof4-eta500"

tst_fname = '{}.log'.format(prefix)
tst = np.loadtxt(tst_fname, dtype=complex)

plt.plot(tst[:, 0], np.abs(tst[:, 1]), '-', label="$P_0$ ({})".format(prefix))
plt.plot(tst[:, 0], np.abs(tst[:, -1]), '-', label="$P_1$ ({})".format(prefix))
plt.plot(tst[:, 0], np.real(tst[:, 2]), '--', label="$\Re r$ ({})".format(prefix))
plt.plot(tst[:, 0], np.imag(tst[:, 2]), '--', label="$\Im r$ ({})".format(prefix))
plt.plot(tst[:, 0], np.abs(tst[:, 2]), '--', label="$|r|$ ({})".format(prefix))

plt.legend(loc=1)
plt.title('Delta model')

plt.savefig('{}.png'.format(prefix))
