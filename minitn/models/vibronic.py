#!/usr/bin/env python
# coding: utf-8
r"""Functions and objects about vibronic coupling model::

    H(R) = H_0(R) + W(R)

where :math:`H_0 = \sum_i \frac{\omega_i}{2}(\pdv[2]{R_i} + R_i^2)I`,. H_B similar (discretized from :math:`J_B`)

References
----------
.. [1] Phys. Chem. Chen. Phys. 16,15957 (2014).
       https://doi.org/10.1039/c4cp02165g
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from itertools import filterfalse, count

from minitn.lib.backend import np
from scipy import linalg
from scipy.integrate import quad

from minitn.lib.tools import __, huffman_tree
from minitn.models.particles import Phonon
from minitn.tensor import Tensor, Leaf
from minitn.algorithms.ml import MultiLayer


class VibronicModel(object):
    ELEC_NAME = 'ELEC'
    VIB_PREFIX = 'VIB'
    ELEC = ['e_list']
    VIB = ['dim_list', 'omega_list', 'kappa_list', 'gamma_list', 'lambda_list']

    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        e_list : [float]
            n  :=  len(e_list) in the following context.
        omega_list : [float]
            m  :=  len(omega_list)
        kappa_list : [[float]]
            (2D) n * m array such that kappa_list[i][j] = 
            :math:`\kappa^{(i)}_j`.
        gamma_list : [[float]]
            (2D) n * m array such that kappa_list[i][j] = 
            :math:`\gamma^{(i)}_j`.
        lambda_list : [[[float]]]
            (3D) n * n * m array such that kappa_list[i][j][k] = 
            :math:`\lambda^{(ij)}_k`.
        """
        valid_attributes = self.ELEC + self.VIB
        for name, value in kwargs.items():
            if name in valid_attributes:
                setattr(self, name, value)
            else:
                logging.warning(__('Parameter {} unexpected, ignored.', name))
        return

    @staticmethod
    def _update(graph, leaves, root, n_branch, prefix='A'):
        subtree, r = huffman_tree(leaves, prefix=prefix)
        try:
            subtree[root] = subtree.pop(r)
            graph.update(subtree)
        except KeyError:
            pass
        return

    @property
    def ground_hamiltonian(self):
        """
        Returns
        -------
        [[(str, ndarray)]]
        """
        h_list = []
        elec_name = self.ELEC_NAME
        zipped = zip(self.dim_list, self.omega_list)
        for n, (dim, omega) in enumerate(zipped):
            ph = Phonon(dim, omega, 1. / omega)
            vib_name = self.VIB_PREFIX + str(n)
            h_list.append([[vib_name, ph.hamiltonian]])

        pass

    @property
    def potential_matrix(self):
        """
        Returns
        -------
        [[(str, ndarray)]]
        """
        pass


if __name__ == '__main__':
    # A model for photoinduced ET reactions in mixed-valence
    # systems in solution at zero temperature.
    from minitn.lib.units import Quantity
    from minitn.tensor import Leaf, Tensor
    from minitn.algorithms.ml import MultiLayer
    from minitn.lib.tools import plt, figure
    logging.basicConfig(format='(In %(module)s)[%(funcName)s] %(message)s', level=logging.INFO)

    raise NotImplementedError

# EOF
