#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import minitn
from scipy.linalg import block_diag


mat = np.array([
    [1,0,0,0],
    [1,1,0,0],
    [1,0,1,0],
    [1,0,0,1]])
q, r = np.linalg.qr(mat)
print(q)
q_c = q[:, 1:]
lam = np.diag((1, 2, 1.3, 1.4))
lam_c = np.dot(q_c.T, np.dot(lam, q_c))
w, v = np.linalg.eigh(lam_c)
v = np.hstack((q[:, :1], np.dot(q_c, v)))
print(w)
lam_f = np.dot(v.T, np.dot(lam, v))

@np.vectorize
def threshold(x):
    return x if np.abs(x) > 1.0e-15 else 0.0

print(threshold(lam_f))
