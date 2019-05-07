#!/usr/bin/env python
# coding: utf-8
r"""Functions and objects about spin-boson model::

    H = H_e + H_v + H_B

where :math:`H_v = 1/2 \sum_j (p_j^2 + \omega_j^2(q_j -\frac{2c_j}{\omega_j^2}
|2><2|)^2)`. H_B similar (discretized from :math:`J_B`)

References
----------
.. [1] J. Chem. Phys. 124, 034114 (2006)
       https://doi.org/10.1063/1.2161178
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip

import numpy as np
from scipy.integrate import quad

from minitn.lib.tools import __
from minitn.models.particles import Phonon
from minitn.models import bath
from minitn.tensor import Tensor, Leaf
from minitn.ml import MultiLayer


class SpinBosonModel(object):
    ELEC = ['e1', 'e2', 'v']
    INNER = ['omega_list', 'lambda_list', 'dim_list']
    OUTER = ['stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d']

    def __init__(self, **kwargs):
        """ Needed parameters:
        - (for electronic part)
            'e1', 'e2', 'v',
        - (for inner part)
            'omega_list', 'lambda_list', 'dim_list',
        - (for outer part)
            'stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d'
        """
        valid_attributes = self.ELEC + self.INNER + self.OUTER
        for name, value in kwargs.items():
            if name in valid_attributes:
                setattr(self, name, value)
            else:
                logging.warning(__('Parameter {} unexpected, ignored.', name))
        self.elec_leaf = None
        self.leaves = []
        h_list = self.electron_hamiltonian()
        h_list.extend(self.inner_hamiltonian() + self.outer_hamiltonian())
        self.h_list = h_list
        return
    
    # TODO: 整理同类项

    def electron_hamiltonian(self, name='ELEC'):
        """
        Returns
        -------
        [(str, ndarray)]
        """
        e1, e2, v = [getattr(self, name) for name in self.ELEC]
        leaf = str(name)
        h = [[e1, v],
             [v, e2]]
        self.elec_leaf = leaf
        self.leaves.append(leaf)
        return [[(leaf, np.array(h))]]

    def vibration_hamiltonian(self, omega_list, coupling_list, dim_list,
                              prefix='V'):
        """
        Returns
        -------
        h_list: [[(str, ndarray)]]
        """
        elec_leaf = self.elec_leaf
        projector = np.array([[0., 0.],
                              [0., 1.]])
        h_list = []
        zipped = zip(dim_list, omega_list, coupling_list)
        for n, (dim, omega, c) in enumerate(zipped):
            ph = Phonon(dim, omega)
            leaf = prefix + str(n)
            # ph part
            h_list.append([(leaf, ph.hamiltonian)])
            # e-ph part
            h_list.append([(leaf, ph.coordinate_operator),
                           (elec_leaf, -2. * c * projector)])
            # e part
            h_list.append([(elec_leaf, 2. * (c / omega) ** 2 * projector)])
        return h_list

    def inner_hamiltonian(self, prefix='I'):
        omega_list, lambda_list, dim_list = [getattr(self, name)
                                             for name in self.INNER]
        coupling_list = list(np.sqrt(np.array(lambda_list) / 2) *
                             np.array(omega_list))
        return self.vibration_hamiltonian(omega_list, coupling_list, dim_list,
                                          prefix=prefix)

    def outer_hamiltonian(self, prefix='O'):
        stop, n, dim, lambda_g, omega_g, lambda_d, omega_d = [
            getattr(self, name) for name in self.OUTER
        ]
        density = bath.bimodal_spectral_density(lambda_g, omega_g,
                                                lambda_d, omega_d)
        zipped = bath.linear_discretization(density, stop, n)
        omega_list, coupling_list = zip(*zipped)
        dim_list = n * [dim]
        return self.vibration_hamiltonian(omega_list, coupling_list, dim_list,
                                          prefix=prefix)


if __name__ == '__main__':
    from minitn.lib.units import Quantity
    sbm = SpinBosonModel(
        e1=0.,
        e2=Quantity(6500, 'cm-1').value_in_au,
        v=Quantity(500, 'cm-1').value_in_au,
        omega_list=[Quantity(2100, 'cm-1').value_in_au,
                    Quantity(650, 'cm-1').value_in_au,
                    Quantity(400, 'cm-1').value_in_au,
                    Quantity(150, 'cm-1').value_in_au],
        lambda_list=([Quantity(750, 'cm-1').value_in_au] * 4),
        dim_list=[10, 14, 20, 30],
        stop=Quantity(6500, 'cm-1').value_in_au,
        n=30,
        dim=30,
        lambda_g=Quantity(2250, 'cm-1').value_in_au,
        omega_g=Quantity(500, 'cm-1').value_in_au,
        lambda_d=Quantity(1250, 'cm-1').value_in_au,
        omega_d=Quantity(50, 'cm-1').value_in_au,
    )
