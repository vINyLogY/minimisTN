#!/usr/bin/env python
# coding: utf-8
r"""ML type of 4d sho
"""
from __future__ import division

import logging
from builtins import filter, map, range, zip
from functools import partial
from time import time

import numpy as np
from scipy import linalg

from minitn.lib.tools import __, time_this
from sho_model import test_2layers


@time_this
def ref(n_spf=10):
    x0, x1, n_dvr, c, dofs = -5., 5., 10, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    exp.settings(cmf_steps=10, ode_method='RK45')
    t1, a1 = zip(
        *exp.autocorr(steps=100, ode_inter=0.01, fast=True, split=False))
    np.save('./data/mmctdh_{}_t'.format(n_spf), t1)
    np.save('./data/mmctdh_{}_a'.format(n_spf), a1)
    return

def exact(n_spf=1):
    x0, x1, n_dvr, c, dofs = -5., 5., 10, 0.5, 4
    exp = test_2layers(x0, x1, n_dvr, n_spf, dofs, c)
    h = exp.dense_hamiltonian()
    w, v = linalg.eigh(h)
    vh = np.transpose(np.conj(v))
    v0 = np.reshape(exp.root.contraction(), -1)
    dot = np.dot
    a = []
    t = []
    for i in range(1000):
        ti = i * 0.01
        exponent = np.exp(-1.0j * w * ti)
        vi = dot(v, dot(np.diag(exponent), dot(vh, v0)))
        a.append(dot(vi, vi))
        t.append(2 * ti)
    np.save('./data/ed_t', t)
    np.save('./data/ed_a', a)


def main():
    logging.root.setLevel(logging.INFO)
    for i in range(1, 5):
        ref(n_spf=i)
    return

exact()
