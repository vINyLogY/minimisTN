#!/usr/bin/env python
# coding: utf-8
r"""Operator data structure

Operator classes for TN type methods.

- SumProdOp:
    Original type of hamiltonian used in (ML-)MCTDH.

- MatProdOp:
    Matrix-Product Operator.
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from contextlib import contextmanager

import numpy as np
from scipy import linalg

from minitn.lib.numerical import compressed_svd
from minitn.lib.tools import __

class Operator(object):
    """Common parts in different types of operators. If directly call this,
    it would be a operator on one degree of freedom.
    """
    def __init__(self, leaf, array):
        """
        Parameters
        ----------
        leaf : Leaf
        array : ndarray
        """
        self._dof = leaf
        self._array = array

    def member_visitor(self):
        yield (self._dof, self._array)

    def __call__(self, wfn, right=False):
        """
        Parameters
        ----------
        wfn : Tensor
        """
        if self._dof in wfn.leaves:
            self._dof.set_array(self._array)
            prev, direction = self._dof[0]
        else:
            pass


    @property
    def leaves(self):
        pass

class ProdOp(Operator):
    pass

class SumProdOp(ProdOp):
    def __init__(self, h_list, f_list=None):
        pass

class MatProdOp(Operator):
    """MPO in DMRG theory.
    """
    pass
