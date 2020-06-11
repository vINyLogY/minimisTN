#!/usr/bin/env python
# coding: utf-8
from matplotlib import rc
rc('font', family='Times New Roman')
rc('text', usetex=True)

from matplotlib import pyplot as plt
import numpy as np


data_label_list = [
    ('ml-data-2dof', '-', 'linear, 2 DOF'),
    ('ml-data-3dof', '-', 'linear, 3 DOF'),
    ('ml-data-4dof', '-', 'linear, 4 DOF'),
    ('ml-data-2dof_const', '--', 'constant, 2 DOF'),
    ('data-3dof_const', '--', 'constant, 3 DOF'),
    ('ml-data-4dof_const', '--', 'constant, 4 DOF')
]
for data_name, pattern, label in data_label_list:
    data = np.loadtxt(data_name + '.txt', dtype=complex)
    # label = data_name.split('-')[-1]
    t = data[:, 0]
    rho10 = data[:, 2]
    # pattern = '-'
    plt.plot(t, np.log(np.abs(rho10)**2 / data[0, 2]**2), pattern, label=label)
    plt.xlabel('Time (fs)')
    plt.ylabel(r'''$\ln |\rho_{10}(t)/\rho_{10}(0)|^2$''')
plt.legend()
plt.show()


