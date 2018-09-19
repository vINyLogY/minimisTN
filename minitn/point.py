#!/usr/bin/env python
# coding: utf-8
r"""Tensor data structure

Interface to numpy.ndarray.

References
----------
.. [1] arXiv:1603.03039v4
"""
from __future__ import absolute_import, division

import logging
from builtins import filter, map, range, zip
from functools import partial

import numpy as np

from minitn.dvr import SineDVR
from minitn.lib.tools import __


class Point(object):
    """
    Attributes
    ----------
    rank
        The rank of the tensor.
    shape
        The shape of the tensor.
    access : {int: (Point, int)}
    """
    def __init__(self, array, leaf, normalize_axis=None):
        r"""
        Parameters
        ----------
        array : ndarray or None
            if array is None, it means that it is a phantom.
        leaf : bool
            0 for branch nodes (states) and 1 for leaves (operators).
        normalize_axis : int, optional
            Assert `Point.partial_contract(point1, normalize_axis,
            point1, normalize_axis, comp=True)` is an identity matrix.
        """
        self.array = array
        self._leaf = leaf
        self._normalize_axis = normalize_axis

        self._access = {}
        self._subtree_matrix = {}

    @property
    def rank(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    def link(self, i, point, j):
        self._access[i] = (point, j)
        point._access[j] = (self, i)
        return

    def subtree_matrix(self, i, proper=True):
        if proper:
            if i in self._access:
                child, j = self._access[i]
                ans = child.subtree_matrix(j, proper=False)
        else:
            if i in self._subtree_matrix:
                return self._subtree_matrix[i]
            if self._leaf:
                ans = self.array
            else:
                tmp = self.array
                phantom_flag = (self._normalize_axis == i)
                for j, (child, k) in self._access.items():
                    if j != i:
                        child_array = child.subtree_matrix(k, proper=False)
                        if child_array is not None:
                            tmp = self.partial_contract(tmp, j, child_array, k)
                            phantom_flag = False
                if phantom_flag:
                    ans = None
                else:
                    ans = self.partial_contract(
                        tmp, i, self.array, i, comp=True
                    )
                self._subtree_matrix[i] = ans
        assert(ans.ndim == 2 and ans.shape[0] == ans.shape[1])
        return ans

    @staticmethod
    def partial_contract(array1, i, array2, j, comp=False):
        shape_1, shape_2 = map(list, (array1.shape, array2.shape))
        n, m = shape_1[i], shape_2[j]
        if not comp:
            composed_index = np.prod(shape_2[:j] + shape_2[j + 1:])
            new_shape = (
                shape_1[:i] + shape_1[i + 1:] + [composed_index]
            )
            array1 = np.moveaxis(array1, i, -1)
            array1 = np.reshape(array1, (-1, n))
            array2 = np.moveaxis(array2, j, 0)
            array2 = np.reshape(array2, (m, -1))
        else:
            array1 = np.moveaxis(array1, i, 0)
            array1 = np.reshape(array1, (n, -1))
            array2 = np.moveaxis(array2, j, -1)
            array2 = np.reshape(array2, (-1, m))
        ans = np.dot(array1, array2)
        if not comp:
            ans = np.reshape(ans, new_shape)
            ans = np.moveaxis(ans, -1, i)
        return ans


# EOF
