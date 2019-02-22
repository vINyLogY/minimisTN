#!/usr/bin/env python
# coding: utf-8
r"""[Deprecated] Tensor data structure

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
    array
        The dense array of the tensor
    rank
        The rank of the tensor.
    shape
        The shape of the tensor.
    _access : {int: (Point, int)}
    """

    def __init__(self, array=None, leaf=False, normalize_axis=None):
        r"""
        Parameters
        ----------
        array : ndarray or None
            if array is None, it means that it is a phantom, which would be
            treated as identity transformation if it is supposed to be a
            matrix.
        leaf : bool
            0 for branch nodes (states) and 1 for leaves (operators).
        normalize_axis : int, optional
            Assert `Point.partial_contract(array, normalize_axis,
            array.conj(), normalize_axis, complement=True)` is an identity
            matrix.
        """
        # Type-check
        if (
            leaf and
            array is not None and
            (array.ndim != 2 or array.shape[0] != array.shape[1])
        ):
            raise TypeError('Invalid Point type.')

        self.array = array
        self.leaf = leaf
        self._normalize_axis = normalize_axis

        self._access = {}

        # Cache
        self._branch = {}
        self._density = {}

    @property
    def rank(self):
        return self.array.ndim

    @property
    def shape(self):
        return self.array.shape

    @property
    def normalize_axis(self):
        return self._normalize_axis

    def renormalization(self, i, root_restrict=True, two_site=False):
        """Renormalization to shift the `proper` root point from `self` to the
        next point.

        Parameters
        ----------
        i : int
        root_restrict : bool
            Whether to restrict the number of `proper` root
            (`self._normalize_axis is None`) to `1`.
        two_site : bool
            Whether to use the two-site algorithm (DMRG2-like) or the one-site
            algorithm (DMRG1).
        """
        point, j = self._access[i]
        if self.root_restrict and self._normalize_axis is not None:
            raise ValueError(
                'This renormalization may break the normalization direction.'
            )
        if two_site:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return

    def link(self, i, point, j, _one_way=False):
        """
        Parameters
        ----------
        i : int
        point : Point
        j : int
        """
        if i >= self.rank:
            raise ValueError('Link index is not smaller than rank.')
        self._access[i] = (point, j)

        if self.linkable():
            raise ValueError('Too much linkages for self Point.')
        if not _one_way:
            point.link(j, self, i, _one_way=True)
        return

    def unlink(self, i, add_phantom=True, _one_way=False):
        point, j = self._access.pop(i)
        if not self.leaf:
            if add_phantom:
                self.link(i, self.phantom, 0)
        if not _one_way:
            point.unlink(j)
        return

    def linkable(self, enumeration=False):
        if not enumeration:
            max_linkage = 1 if self.leaf else self.rank
            return True if max_linkage < len(self._access) else False
        else:
            indice = (
                i for i in range(self.rank) if i not in self._access
            )
            return indice

    def children(self, parent=None):
        """
        Parameters
        ----------
        parent : None or int or Point, optional
        """
        for i, (point, j) in self._access.items():
            if i != parent and point is not parent:
                yield (i, point, j)

    def branch(self, i, proper=True, phantom_leaves=False):
        """
        Parameters
        ----------
        i : int
            The `axis` of the so-called branch matrix. This direction will be
            treated as root direction and others will be as children.
        proper : bool, optional
            If True then branch does not include the `self` point,
            otherwise it contain the `self` point. Default is True.
        phantom_leaves : bool, optional
            Whether treat all leaf nodes as phantoms. Default is False.

        Returns
        -------
        ans : (n, n) ndarray or None
            A matrix or a phantom.
        """
        if proper:
            if i in self._access:
                child, j = self._access[i]
                ans = child.branch(
                    j, proper=False, phantom_leaves=phantom_leaves
                )
            else:
                raise RuntimeError('Incomplete linkage infomation.')
        else:
            # Load cache
            cache = self._density if phantom_leaves else self._branch
            if i in cache:
                return cache[i]

            # Calculation order
            # 1. For leaf nodes (operators), return its array as answer matrix
            if self.leaf:
                ans = None if phantom_leaves else self.array
            # 2. For other nodes (states), contract its array with the arrays
            # of its children
            else:
                present_array = self.array
                # Phantom: if a matrix is assigned as `None`
                # Treat phantom matrix as identity matrix
                phantom_flag = True
                # Iteration:
                if len(self._access) < present_array.ndim:
                    raise RuntimeError('Incomplete linkage infomation.')
                for j, child, k in self.children(parent=i):
                    child_array = child.branch(
                        k, proper=False, phantom_leaves=phantom_leaves
                    )
                    # Only contract with those that are not phantom
                    if child_array is not None:
                        present_array = self.partial_contract(
                            present_array, j, child_array, k
                        )
                        if phantom_flag:
                            phantom_flag = False
                # If the array only contract with phantoms AND the
                # `partial_contract` is along the `normalize_axis`, then the
                # answer would also be a phantom.
                if phantom_flag and self._normalize_axis == i:
                    ans = None
                else:
                    conj_array = np.conj(self.array)
                    ans = self.partial_contract(
                        present_array, i, conj_array, i, complement=True
                    )
                # Save cache
                cache[i] = ans
        # Type-check
        assert(
            ans is None or
            (ans.ndim == 2 and ans.shape[0] == ans.shape[1])
        )
        return ans

    def density(self, axis=None):
        if axis is None:
            axis = self._normalize_axis
            if axis is None:
                raise ValueError('Need to specify the density axis.')

        ans = self.branch(axis, phantom_leaves=True)
        return ans

    def projection(self, axis=None):
        if axis is None:
            axis = self._normalize_axis
            if axis is None:
                raise ValueError('Need to specify the projection axis.')

        array = self.array
        conj_array = np.conj(self.array)
        ans = self.partial_contract(array, axis, conj_array, axis)
        return ans

    def visitor(self, parent=None, with_leaves=True):
        """A recursive generator traveling the tree from the self point.
        """
        for _, child, _ in self.children(parent=parent):
            for point in child.visitor(parent=self):
                if with_leaves:
                    yield point
                else:
                    if not self.leaf:
                        yield point
        yield self

    def autocomplete(self):
        """Auto complete the tree with phantom nodes. Start from `self`.
        """
        for point in instance.visitor():
            for i in point.linkable(enumeration=True):
                new_phantom = self.phantom()
                point.link(i, new_phantom, 0)
        return

    @classmethod
    def phantom(cls):
        return cls(array=None, leaf=True)

    @staticmethod
    def partial_contract(array1, i, array2, j, complement=False):
        shape_1, shape_2 = map(list, (array1.shape, array2.shape))
        n, m = shape_1[i], shape_2[j]
        if not complement:
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
        if not complement:
            ans = np.reshape(ans, new_shape)
            ans = np.moveaxis(ans, -1, i)
        return ans

# EOF
