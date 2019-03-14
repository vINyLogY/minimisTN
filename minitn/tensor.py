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
from itertools import chain

import numpy as np
from scipy import linalg

from minitn.lib.tools import __
from minitn.lib.numerical import compressed_svd


_empty = object()


class Tensor(object):
    r"""
    Attributes
    ----------
    array : ndarray, None
        The dense array of the tensor.
    order : int, None
        The order(/rank/etc.) of the tensor.
    shape : [int], None
        The shape of the tensor.
    axis : int
        The direction of normalization.
    name : str
    aux : ndarray, None
        For anything
    _access : {int: (Point, int)}
    _partial_env : {int: 2-d ndarray}
    """

    def __init__(self, name=None, array=None, axis=None):
        r"""
        Parameters
        ----------
        name : str
        array : ndarray
        axis : int
            Assert `partial_trace(array, axis, array.conj(), axis)` is an
            identity matrix.
        """
        self.name = str(name)
        self.axis = axis

        self.set_array(array)
        self._access = {}

        # Cache
        self._partial_env = {}    # {(int, matrix)}, saving mean field

        # For some special methods
        self.aux = None
        return

    def __str__(self):
        string = self.name
        return 'Tensor_' + string if self.name is not None else repr(self)

    @staticmethod
    def generate(graph, root):
        """Helper function to generate a graph with Tensors as nodes.

        Parameters
        ----------
        graph : {string: [string]}
        """
        name = root
        root = Tensor(name=name, axis=None)
        order = {name: 0}    # {string: int}, which means {tensor: order}
        stack = [(name, root)]
        while stack:
            sr, r = stack.pop()
            for sc in graph[sr]:
                if sc in graph:
                    c = Tensor(name=sc, axis=0)
                else:
                    c = Leaf(name=sc)
                r.link_to(order[sr], c, 0)
                order[sr] += 1
                order[sc] = 1
                if sc in graph:
                    stack.append((sc, c))
        return root

    def set_array(self, array):
        self._array = (
            np.array(array, dtype='complex128') if array is not None else None
        )
        return

    def reset(self):
        self._partial_env = {}

    @property
    def array(self):
        if self._array is None:
            raise RuntimeError('No specific array set at {0}!'.format(self))
        else:
            return np.array(self._array, dtype='complex128')    # Return a copy

    @property
    def shape(self):
        if self._array is None:
            return None
        else:
            return list(self._array.shape)    # Return a copy

    @property
    def order(self):
        if self._array is None:
            return None
        else:
            return len(self._array.shape)

    @staticmethod
    def link(a, i, b, j):
        r"""Add a linkage info between i-th order of a and j-th order of b.

        Parameters
        ----------
        a : Tensor
        i : int
        b : Tensor
        j : int
        """
        # Logging info
        for k, tensor in ((i, a), (j, b)):
            if k in tensor._access:
                logging.info(__(
                    'Overwrite the {0}-th linkage info of {1}',
                    k, tensor
                ))

        a._access[i] = (b, j)
        b._access[j] = (a, i)
        return

    @staticmethod
    def unlink(a, i, b, j):
        r"""Remove a linkage info between i-th order of a and j-th order of b.

        Parameters
        ----------
        a : Tensor
        i : int
        b : Tensor
        j : int
        """
        if a._access[i] == (b, j) and b._access[j] == (a, i):
            a._access.pop(i)
            b._access.pop(j)
        else:
            raise RuntimeError('No such linkage.')
        return

    def check_linkage(self, i, strict=False):
        """
        Parameters
        ----------
        i : int
        strict : bool
            Whether to check the compatibility of arrays.
        """
        child, j = self._access[i]    # raise KeyError if no such linkage
        condition = (child._access[j] == (self, i))
        if strict:
            condition = condition and (
                (isinstance(self, Leaf) and self._array is None) or
                (isinstance(child, Leaf) and child._array is None) or (
                    not (self._array is None or child._array is None) and
                    self.shape[i] == child.shape[j]
                )
            )
        if condition:
            logging.debug(__(
                'Checked {0}-th linkage info of {1}{2}',
                i, self, ' (strict)' if strict else ''
            ))
            return
        else:
            raise RuntimeError(
                'Wrong {0}-th linkage info of {1}'.format(i, self)
            )

    def check_completness(self, strict=False):
        """
        Parameters
        ----------
        strict : bool
            Whether to check the compatibility of arrays.
        """
        if self.order is None:
            raise RuntimeError('No specific array at {0}'.format(self))
        elif set(self._access) != set(range(self.order)):
            raise RuntimeError('Incomplete linkages info at {0}'.format(self))
        else:
            for i in range(self.order):
                self.check_linkage(i, strict=strict)
            logging.info(__(
                'Checked linkage completeness of {0}{1}',
                self, ' (strict)' if strict else ''
            ))
            return

    def link_to(self, i, b, j):
        Tensor.link(self, i, b, j)
        return

    def unlink_to(self, i, fast=False):
        """
        Return
        ------
        child : Tensor
        j : int
        """
        try:
            if not fast:
                self.check_linkage(i)
        except KeyError:
            logging.info(__(
                'No {0}-th linkage info of {1}', i, self
            ))
        else:
            child, j = self._access[i]
            Tensor.unlink(self, i, child, j)
        return child, j

    def children(self, axis=_empty, leaf=True):
        """Generator which yields the children of self.
        Yield
        -----
        tuple : (int, Tensor, int)
        """
        def key(x): return x[0]
        if axis is _empty:
            axis = self.axis
        for i, (tensor, j) in sorted(self._access.items(), key=key):
            if axis is None or i != axis:
                if leaf or not isinstance(tensor, Leaf):
                    yield (i, tensor, j)

    def visitor(self, axis=_empty, leaf=True):
        """
        Yield
        -----
        tensor : Tensor
        """
        if leaf or not isinstance(self, Leaf):
            yield self
        for _, child, j in self.children(axis=axis, leaf=leaf):
            for tensor in child.visitor(axis=j, leaf=leaf):
                yield tensor

    def linkage_visitor(self, axis=_empty, leaf=True, back=False):
        """
        Yield
        -----
        tuple : (Tensor, int, Tensor, int)
        """
        for i, child, j in self.children(axis=axis, leaf=leaf):
            yield (self, i, child, j)
            for linkage in child.linkage_visitor(axis=j, leaf=leaf, back=back):
                yield linkage
            if back:
                yield (child, j, self, i)

    def iddfs_visitor(self, axis=_empty, leaf=False, back=True):
        pass

    def partial_env(self, i, proper=False, use_aux=False):
        """
        Parameters
        ----------
        i : {int, None}
        proper : bool
        use_aux : bool
            Whether to use self.aux as conj.
        Returns
        -------
        ans : {2-d ndarray, None}
        """
        if proper:    # Only calculate non-proper subtree directly
                      # to support the Leaf
            child, j = self._access[i]
            return child.partial_env(j, proper=False, use_aux=use_aux)

        else:
            partial_product = Tensor.partial_product
            partial_trace = Tensor.partial_trace
            # Check the cache
            if i in self._partial_env and not use_aux:
                return self._partial_env[i]
            # Main algorithm
            else:
                env_ = [
                    (i_, tensor.partial_env(j, proper=False, use_aux=use_aux))
                    for i_, tensor, j in self.children(axis=i)
                ]    # Recursively
                # Make use of the normalization condition
                if (
                    i == self.axis and
                    not (True for _, matrix in env_ if matrix is not None)
                ):
                    ans = None
                else:
                    temp = self.array
                    for i_, matrix in env_:
                        temp = partial_product(temp, i_, matrix)
                    conj = self.aux if use_aux else np.conj(self.array)
                    ans = partial_trace(temp, i, conj, i)
                # Cache the answer and return
                if i is not None:
                    if not use_aux:
                        self._partial_env[i] = ans
                else:
                    ans = ans if ans is not None else 1.
                return ans

    def global_inner_product(self):
        """Return <aux*|array> (NO conj)
        """
        for leaf in self.leaves():
            leaf.aux = None
        return self.partial_env(None, use_aux=True)

    def matrix_element(self):
        """Return <aux*|H|array> (NO conj)
        """
        for leaf in self.leaves():
            leaf.aux = leaf.array
        return self.partial_env(None, use_aux=True)

    def global_norm(self):
        """Return <array|array>
        """
        for t in self.visitor(leaf=False):
            t.aux = np.conj(t.array)
        ans = self.global_inner_product()
        return np.sqrt(ans)

    def expection(self):
        """Return <array|H|array>
        """
        for t in self.visitor(leaf=False):
            t.aux = np.conj(t.array)
        return self.matrix_element()

    def local_inner_product(self):
        """No conj
        """
        a, b, i = self.array, self.aux, self.axis
        return Tensor.partial_trace(a, i, b, i)

    def local_norm(self):
        self.aux = np.conj(self.array)
        ans = self.local_inner_product()
        return np.sqrt(ans)

    def leaves(self):
        """
        Returns
        -------
        [Leaf]
        """
        ans = []
        for tensor in self.visitor(leaf=True):
            if isinstance(tensor, Leaf):
                ans.append(tensor)
        return ans

    def vectorize(self, use_aux=False):
        vec_list = []
        for t in self.visitor(leaf=False):
            array = t.aux if use_aux else t.array
            vec = np.reshape(array, -1)
            vec_list.append(vec)
        ans = np.concatenate(vec_list, axis=None)
        return ans

    def tensorize(self, vec, use_aux=False):
        start = 0
        for t in self.visitor(leaf=False):
            shape = t.shape
            end = start + np.prod(shape)
            array = np.reshape(vec[start:end], shape)
            if use_aux:
                t.aux = array
            else:
                t.set_array(array)
            start = end
        return

    def split(self, axis, indice=None, root=None, child=None,
              rank=None, err=None):
        """Split the root Tensor to a certain axis/certain axes.

        Parameters
        ----------
        axis : {int, [int]}
        rank : int
            Max rank in SVD
        err : float
            Max error in SVD
        indice : (int, int)
            Linkage between root and child
        root : Tensor
            Tensor to be a new root node. `None` to create a new Tensor.
        child : Tensor
            Tensor to be a new child node. `None` to create a new Tensor.

        Return
        ------
        root : Tensor
            New root node in the same environment of self.
        child : Tensor
            New child node in the same environment of self.
        """
        if self.axis is not None:
            raise RuntimeError('Can only split the root Tensor!')
        try:
            axes1 = list(sorted(axis))
        except TypeError:
            axes1 = [axis]

        # save all data needed in `self` (may be reinitialized)
        a = self.array
        axes2 = list(range(self.order))
        for i in axes1:
            axes2.remove(i)
        children = list(self.children(axis=None))
        cls = type(self)
        shape1 = self.shape
        name = self.name
        index1, index2 = indice if indice is not None else (0, 0)
        #
        if '+' in name:
            name1, name2 = name.split('+')
        else:
            name1, name2 = name, name + '\''

        shape2 = []
        for i in axes2:
            shape2.append(shape1.pop(i))
            a = np.moveaxis(a, i, -1)
        a = np.reshape(a, (np.prod(shape1), np.prod(shape2)))
        u, s, vh = compressed_svd(a, rank=rank, err=err)
        root_array = np.reshape(np.dot(u, s), shape1 + [-1])
        root_array = np.moveaxis(root_array, -1, index1)
        child_array = np.reshape(vh, [-1] + shape2)
        child_array = np.moveaxis(child_array, 0, index2)
        if root is None:
            root = cls(name=name1, array=root_array, axis=None)
        else:
            root.name, root.axis = name1, None
            root.set_array(root_array)
        if child is None:
            child = cls(name=name2, array=child_array, axis=0)
        else:
            child.name, child.axis = name2, index2
            child.set_array(child_array)
        # Fix linkage info
        unlink = Tensor.unlink
        link = Tensor.link
        link_info = []
        for i, t, j in children:
            is_1 = i in axes1
            index_ = index1 if is_1 else index2
            axes = axes1 if is_1 else axes2
            tensor = root if is_1 else child
            unlink(self, i, t, j)
            ax = axes.index(i)
            if ax >= index_:
                ax += 1
            link_info.append((tensor, ax, t, j))
        for linkage in link_info:
            link(*linkage)
        link(root, index1, child, index2)
        return root, child

    def unite(self, axis, root=None):
        """Unite a Tensor `t` on a certain axis with self.

        Parameters
        ----------
        axis : int
        root : Tensor
            Tensor to be a new root node. `None` to create a new Tensor.

        Return
        ------
        root : Tensor
            New root node in the same environment of self.
        """
        cls = type(self)
        t1 = self
        t2, i = t1.unlink_to(axis, fast=True)
        if t1.axis is not None and t2.axis is not None:
            raise RuntimeError(
                'Can only unite root Tensor with another Tensor!'
            )
        axes1 = list(range(t1.order))
        axes1.remove(axis)
        axes2 = list(range(t2.order))
        axes2.remove(i)
        shape1, shape2 = t1.shape, t2.shape
        shape1.pop(axis)
        shape2.pop(i)
        a1, a2 = t1.array, t2.array
        array = Tensor.partial_product(a1, axis, a2, i)
        array = np.moveaxis(array, axis, -1)
        array = np.reshape(array, shape1 + shape2)
        name = t1.name + '+' + t2.name
        if root is None:
            root = cls(name=name, array=array, axis=None)
        else:
            root.name, root.axis = name, None
            root.set_array(array)

        # Fix linkage info
        link = Tensor.link
        unlink = Tensor.unlink
        children1 = list(t1.children(axis=None))
        children2 = list(t2.children(axis=None))
        link_info = []
        for j, t, k in children1:
            unlink(t1, j, t, k)
            link_info.append((root, axes1.index(j), t, k))
        for j, t, k in children2:
            unlink(t2, j, t, k)
            link_info.append((root, axes2.index(j) + len(axes1), t, k))
        for linkage in link_info:
            link(*linkage)
        return root

    def normalize(self):
        array = self.array
        norm = np.array(self.local_norm())
        if norm.ndim == 0:
            self.set_array(array / norm)
        elif norm.ndim == 2:
            rank = np.linalg.matrix_rank(array)
            mean_norm = np.trace(norm) / rank
            self.set_array(array / mean_norm )
        else:
            raise RuntimeError()

    def projector(self, comp=False):
        """[Deprecated] Return the projector corresponding to self.

        Returns
        -------
        ans : ndarray
        """
        axis = self.axis
        if axis is not None:
            array = self.array
            shape = self.shape
            dim = shape.pop(self.axis)
            comp_dim = np.prod(shape)
            array = np.moveaxis(array, axis, -1)
            array = np.reshape(array, (-1, dim))
            array_h = np.conj(np.transpose(array))
            ans = np.dot(array, array_h)
            if comp:
                identity = np.identity(comp_dim)
                ans = identity - ans
            ans = np.reshape(ans, shape * 2)
            return ans
        else:
            raise RuntimeError('Need to specific the normalization axis!')

    @staticmethod
    def partial_product(array1, i, array2, j=0):
        r"""Times a matrix to a tensor.

               |
            -- 1 -i--j- 2 --
               |

        Parameters
        ----------
        array1 : ndarray
        i : int
        array2 : {2-d ndarray, None}
            None means doing nothing
        j : {0, 1}
            Default: 0

        Returns
        -------
        ans : ndarray
        """
        if array2 is None:
            return array1
        else:
            return Tensor._partial_product(array1, i, array2, j)

    @staticmethod
    def _partial_product(array1, i, array2, j):
        r"""Times a matrix to a tensor.

               |
            -- 1 -i--j- 2 --
               |

        Parameters
        ----------
        array1 : ndarray
        i : int
        array2 : 2-d ndarray 
        j : int

        Returns
        -------
        ans : ndarray
        """
        shape_1, shape_2 = map(list, (array1.shape, array2.shape))
        n, m = shape_1.pop(i), shape_2.pop(j)
        new_shape = shape_1 + [np.prod(shape_2)]
        array1 = np.moveaxis(array1, i, -1)
        array1 = np.reshape(array1, (-1, n))
        array2 = np.moveaxis(array2, j, 0)
        array2 = np.reshape(array2, (m, -1))
        ans = np.dot(array1, array2)
        ans = np.reshape(ans, new_shape)
        ans = np.moveaxis(ans, -1, i)
        return ans

    @staticmethod
    def partial_trace(array1, i, array2, j):
        r"""Partial trace of 2 tensors, return a matrix. 

                +----+
                |    |
            -i- 1 -- 2 -j-

        Parameters
        ----------
        array1 : ndarray
        i : {int, None}
            if i is None then j must be None.
        array2 : ndarray 
        j : {int, None}
            if j is None then i must be None.

        Returns
        -------
        ans : ndarray
            Of shape `(n, m)`
        """
        if i is not None and j is not None:
            shape_1, shape_2 = map(list, (array1.shape, array2.shape))
            n, m = shape_1[i], shape_2[j]
            array1 = np.moveaxis(array1, i, 0)
            array1 = np.reshape(array1, (n, -1))
            array2 = np.moveaxis(array2, j, -1)
            array2 = np.reshape(array2, (-1, m))
        elif i is None and j is None:
            array1 = np.reshape(array1, -1)
            array2 = np.reshape(array2, -1)
        else:
            raise TypeError('Invalid parameters i={} and j={}!'.format(i, j))
        ans = np.dot(array1, array2)
        return ans


class Leaf(Tensor):
    r"""
    Attributes
    ----------
    array : ndarray, None
        The dense array of the tensor. `None` means identity matrix.
    order : int
        = 1.
    shape : (int, int)
        The shape of the local Hamiltonian.
    name : str
    aux : ndarray, None
    _access : {0: (Tensor, int)}
    """

    def __init__(self, name=None, array=None):
        super(Leaf, self).__init__(name=name, array=array, axis=None)
        return

    def __str__(self):
        string = self.name
        return 'Leaf_' + string if self.name is not None else repr(self)

    @property
    def array(self):
        ans = self._array
        return ans if ans is None else np.array(ans, dtype='complex128')

    order = 1    # Treat as an end point in a tensor tree

    def reset(self):
        self._array = None

    def partial_env(self, i, proper=False, use_aux=False):
        """
        Returns
        -------
        ans : 2-d ndarray
        """
        if proper:
            raise RuntimeError('No proper mean field at Leaf {}'.format(self))
        elif i != 0:
            raise RuntimeError('For Leaf {} `i` must be 0'.format(self))
        else:
            return self.aux if use_aux else self.array

# EOF
