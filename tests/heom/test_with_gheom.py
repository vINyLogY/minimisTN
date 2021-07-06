#!/usr/bin/env python3
# coding: utf-8
from __future__ import absolute_import, division, print_function

import logging
from numpy.linalg import eig

from scipy.integrate._ivp.rk import RK45
from minitn.tensor import Leaf
import os
import sys
import pytest
from builtins import filter, map, range, zip

import numpy as np
from scipy import linalg

from minitn.heom.noise import Correlation, Drude
from minitn.lib.units import Quantity
from minitn.heom.eom import Hierachy
from minitn.heom.propagate import MultiLayer
from minitn.structures.states import Tensor
from minitn.lib.logging import Logger

f_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(f_dir, 'gheom'))


def test_drude():
    from minitn.heom.noise import Drude
    from minitn.lib.units import Quantity

    # System
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au

    # Bath
    corr2 = Correlation(k_max=1)
    corr2.symm_coeff  = [0]  # [4.66691921e+01 * 9.24899189e+01]
    corr2.asymm_coeff = [0]  # [4.66691921e+01 * -2.35486582e+01]
    corr2.exp_coeff = [1.52764702e+02]
    corr2.delta_coeff = 0 # delta_coeff()
    corr2.print()

    h = np.array([[e, v],
                  [v, 0]])

    op = np.array([[1, 0],
                   [0, -1]])


    # Superparameters
    max_tier  = 10 # (number of possble values for each n_k in the extended rho)

    heom = Hierachy([max_tier], h, op, corr2)

    phi = np.array([1, 0]) 
    rho_0 = np.tensordot(phi, phi, axes=0)
    init_rho = heom.gen_extended_rho(rho_0)

    # Simple HEOM
    root = Tensor(name='root', array=init_rho, axis=None)
    for k in range(1 + 2):
        name_str = k
        l = Leaf(name=name_str)
        root[k] = (l, 1)

    solver = MultiLayer(root.array, heom.diff())

    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=20000,
        ode_inter=0.1,
    )):
        if n % 100 == 0:
            rho = np.reshape(r, (-1, 4))
            for n, _ in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
                    print('Time: {}; rho: {}  {}  {}  {}'.format(*flat_data))
    return np.array(dat)


def test_drude():
    from minitn.heom.noise import Drude
    from minitn.lib.units import Quantity

    # System
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au


    # Bath
    # 5 cm-1 and 0.02
    corr = Drude(lambda_=2.2782e-5, omega_0=4.8378e-5, k_max=1, beta=1)
    corr.print()

    h = np.array([[e, v],
                  [v, 0]])

    op = np.array([[1, 0],
                   [0, -1]])


    # Superparameters
    max_tier  = 10 # (number of possble values for each n_k in the extended rho)

    heom = Hierachy([max_tier], h, op, corr)

    phi = np.array([1, 0]) 
    rho_0 = np.tensordot(phi, phi, axes=0)
    init_rho = heom.gen_extended_rho(rho_0)

    # Simple HEOM
    root = Tensor(name='root', array=init_rho, axis=None)
    for k in range(1 + 2):
        name_str = k
        l = Leaf(name=name_str)
        root[k] = (l, 1)

    solver = MultiLayer(root.array, heom.diff())

    # Define the obersevable of interest
    dat = []
    for n, (time, r) in enumerate(solver.propagator(
        steps=20000,
        ode_inter=0.1,
    )):
        if n % 100 == 0:
            rho = np.reshape(r, (-1, 4))
            for n, _ in enumerate(rho):
                if n == 0:
                    flat_data = [time] + list(rho[0])
                    dat.append(flat_data)
                    print('Time: {}; rho: {}  {}  {}  {}'.format(*flat_data))
    return np.array(dat)


def test_uni():
    from scipy.integrate import solve_ivp
    # System
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au

    h = np.array([[e, v],
                  [v, 0]], dtype='double')

    phi = np.array([1, 0], dtype='complex') 
    ode_int = 0.1
    diff = lambda t, y: (h @ y) * (-1.0j)
    
    for n in range(20000):
        time  =  ode_int * n
        solver = solve_ivp(
            diff, (time, time + ode_int), phi, method='RK23', max_step=ode_int,
            rtol=1e-8, atol=1e-8
        )
        phi = solver.y[:, -1]
        if n % 100 == 0:
            print('Time: {}; rho: {}    {}'.format(time, np.abs(phi[0])**2, np.abs(phi[1])**2))

def test_uni_int():
    from scipy.integrate import solve_ivp
    from scipy.linalg import eigh
    # System
    e = Quantity(100, 'cm-1').value_in_au
    v = Quantity(50, 'cm-1').value_in_au

    h = np.array([[e, v],
                  [v, 0]], dtype='double')

    l, u = eigh(h)
    uc = np.conj(np.transpose(u))

    phi = np.array([1, 0], dtype='complex') 
    ode_int = 0.1
    
    for n in range(20000):
        time  =  ode_int * n
        vec =  np.dot(uc, (np.exp(-1.0j * time * l) * np.dot(u, phi)))
        if n % 100 == 0:
            print('Time: {}; rho: {}    {}    ({})'.format(time, np.abs(vec[0])**2, np.abs(vec[1])**2, np.linalg.norm(vec)))



if __name__ == '__main__':
    test_drude()
    #test_uni_int()

