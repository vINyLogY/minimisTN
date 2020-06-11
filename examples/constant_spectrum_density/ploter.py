#!/usr/bin/env python
# coding: utf-8
from matplotlib import rc
rc('font', family='Times New Roman')
rc('text', usetex=True)

from matplotlib import pyplot as plt
import numpy as np


# data_list2 = [
#     'data-1dof-500cutoff',
#     'data-2dof-1000cutoff',
#     'data-3dof-1500cutoff',
# ]
# data_list3 = [
#     'data-1000cutoff',
#     'data-2000cutoff',
#     'data-3000cutoff',
#     'data-4000cutoff',
# ]
# data_list1 = [
#     'data-1dof',
#     'data-2dof',
#     'data-3dof',
# ]

data_list_ml = [
    'data-1dof-500cutoff',
    'data-2dof-1000cutoff',
    'data-3dof-1500cutoff',
#    'ml-data-4dof-2000cutoff',
]
data_list_ml2 = [
    'data-1dof',
    'data-2dof',
    'data-3dof',
    'ml-data-4dof',
    'ml-data-8dof',
#    'ml-data-4dof-2000cutoff',
]
for data_name in data_list_ml2:
    data = np.loadtxt(data_name + '.txt', dtype=complex)
    label = data_name.split('-')[-1]
    t = data[:, 0]
    rho10 = data[:, 2]
    pattern = '-'
    plt.plot(t, np.log(np.abs(rho10)**2 / data[0, 2]**2), pattern, label=label)
    plt.xlabel('Time')
    plt.ylabel(r'''$|\rho_{10}(t)/\rho_{10}(0)|^2$''')
plt.legend()
plt.show()


