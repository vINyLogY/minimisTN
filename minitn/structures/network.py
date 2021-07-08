# #!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import, division, print_function
from minitn.lib.backend import np
from minitn.lib.numerical import triangular


def autocomplete(root, n_bond_dict):
    """Autocomplete the tensors linked to `self.root` with suitable initial
    value.

    Parameters
    ----------
    root : Tensor
    n_bond_dict : {Leaf: int}
        A dictionary to specify the dimensions of each primary basis.
    """
    for t in root.visitor(leaf=False):
        if t.array is None:
            axis = t.axis
            n_children = []
            for i, child, j in t.children():
                n_children.append(n_bond_dict[(t, i, child, j)])
            if axis is not None:
                p, p_i = t[axis]
                n_parent = n_bond_dict[(p, p_i, t, axis)]
                shape = [n_parent] + n_children
            else:
                n_parent = 1
                shape = n_children
            array = np.zeros((n_parent, np.prod(n_children)))
            for n, v_i in zip(triangular(n_children), array):
                v_i[n] = 1.
            array = np.reshape(array, shape)
            if axis is not None:
                array = np.moveaxis(array, 0, axis)
            t.set_array(array)
            t.normalize(forced=True)
            assert (
                t.axis is None or
                np.linalg.matrix_rank(t.local_norm()) == t.shape[t.axis]
            )
    if __debug__:
        for t in root.visitor():
            t.check_completness(strict=True)
    return
