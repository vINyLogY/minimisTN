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
from itertools import filterfalse, count

import numpy as np
from scipy import linalg
from scipy.integrate import quad

from minitn.lib.tools import __, huffman_tree
from minitn.models.particles import Phonon
from minitn.models import bath
from minitn.tensor import Tensor, Leaf
from minitn.algorithms.ml import MultiLayer


# TODO: need to be polished
class SpinBosonModel(object):
    ELEC = ['e1', 'e2', 'v']
    INNER = ['omega_list', 'lambda_list', 'dim_list']
    OUTER = ['stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d']
    FIELD = ['mu', 'tau', 't_d', 'omega']

    def __init__(self, including_bath=False, relaxation_list=None,
                 **kwargs):
        """ Needed parameters:
        - (for electronic part)
            'e1', 'e2', 'v',
        - (for inner part)
            'omega_list', 'lambda_list', 'dim_list',
        - (for outer part)
            'stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d'
        """
        # FIXME: including_bath=True case
        valid_attributes = self.ELEC + self.INNER + self.OUTER + self.FIELD
        for name, value in kwargs.items():
            if name in valid_attributes:
                setattr(self, name, value)
            else:
                logging.warning(__('Parameter {} unexpected, ignored.', name))
        self.elec_leaf = None
        self.inner_prefix = None
        self.outer_prefix = None
        self.leaves = []
        self.dimensions = {}
        self.including_bath = including_bath
        h_list = self.electron_hamiltonian()
        all_vibriations = (self.inner_hamiltonian() + self.outer_hamiltonian()
                           if including_bath else self.inner_hamiltonian())
        h_list.extend(all_vibriations)
        if relaxation_list is not None:
            h_list.extend(
                self.relaxation_hamiltonian(
                    coupling_list=relaxation_list, prefix='I')
            )
        self.h_list, self.f_list = self.collect_electric_terms(h_list)
        return

    def autograph_mctdh(self):
        if self.including_bath:
            raise NotImplementedError()
        leaves = self.inner_leaves
        spfs = [s + "s" for s in leaves]
        graph = {'ROOT': [self.elec_leaf] + spfs}
        graph.update(dict(zip(spfs, leaves)))
        return graph, 'ROOT'

    def autograph_full(self):
        if self.including_bath:
            raise NotImplementedError()
        leaves = self.inner_leaves
        graph = {'ROOT': [self.elec_leaf] + leaves}
        return graph, 'ROOT'

    def autograph(self, n_branch=2):
        graph = {'ROOT': [self.elec_leaf, 'INNER', 'OUTER']}
        if self.including_bath:
            self._update(graph, self.inner_leaves, 'INNER', n_branch,
                         prefix='AI')
            self._update(graph, self.outer_leaves, 'OUTER', n_branch,
                         prefix='AO')
        else:
            mid = len(self.inner_leaves) // 2
            self._update(graph, self.inner_leaves[:mid], 'INNER', n_branch,
                         prefix='AI1')
            self._update(graph, self.inner_leaves[mid:], 'OUTER', n_branch,
                         prefix='AI2')
        return graph, 'ROOT'

    def autograph_with_aux(self, n_branch=2):
        graph = {
            'ROOT': ['ELECs', 'INNER', 'OUTER'],
        }
        if self.including_bath:
            inner_spfs = [name + 's' for name in self.inner_leaves]
            outer_spfs = [name + 's' for name in self.outer_leaves]
        else:
            mid = len(self.inner_leaves) // 2
            inner_spfs = [name + 's' for name in self.inner_leaves[:mid]]
            outer_spfs = [name + 's' for name in self.inner_leaves[mid:]]
        self._update(graph, inner_spfs, 'INNER', n_branch, prefix='AI')
        self._update(graph, outer_spfs, 'OUTER', n_branch, prefix='AO')
        leaves = self.leaves
        spfs = [name + 's' for name in leaves]
        aux_leaves = [name + "'" for name in leaves]
        aux = {s: [p, q] for s, p, q in zip(spfs, leaves, aux_leaves)}
        graph.update(aux)
        self.dimensions.update({name: self.dimensions[name[:-1]]
                                for name in aux_leaves})
        return graph, 'ROOT'

    @staticmethod
    def _update(graph, leaves, root, n_branch, prefix='A'):
        def obj_new(x=0):
            x += 1
            return str(prefix) + str(x)

        subtree, r = huffman_tree(leaves, obj_new=obj_new)
        try:
            subtree[root] = subtree.pop(r)
            graph.update(subtree)
        except KeyError:
            pass
        return

    def collect_electric_terms(self, h_list, absorbed=False):
        elec_leaf = self.elec_leaf

        def condition(term): return len(term) == 1 and term[0][0] == elec_leaf
        elec_list = filter(condition, h_list)
        elec_array = sum([term[0][1] for term in elec_list])
        left_list = list(filterfalse(condition, h_list))
        try:
            if absorbed:
                field = self.td_electron_hamiltionian(ti_array=elec_array)
                h_list = left_list
                f_list = [[[elec_leaf, field]]]
            else:
                field = self.td_electron_hamiltionian()
                h_list = [[[elec_leaf, elec_array]]] + left_list
                f_list = [[[elec_leaf, field]]]
        except AttributeError:
            h_list = [[[elec_leaf, elec_array]]] + left_list
            f_list = None
        return h_list, f_list

    def td_electron_hamiltionian(self, ti_array=None):
        def field(t):
            mu, tau, t_d, omega = [getattr(self, name) for name in self.FIELD]
            h = [[0., mu],
                 [mu, 0.]]
            delta = t - t_d
            coeff = (np.exp(-4. * np.log(2.) * (delta / tau) ** 2) *
                     np.cos(omega * delta))
            ans = coeff * np.array(h)
            if ti_array is not None:
                ans += ti_array
            return ans
        return field

    def electron_hamiltonian(self, name='ELEC'):
        """
        Returns
        -------
        [(str, ndarray)]
        """
        e1, e2, v = [getattr(self, n) for n in self.ELEC]
        leaf = str(name)
        h = [[e1, v],
             [v, e2]]
        self.elec_leaf = leaf
        self.leaves.append(leaf)
        self.dimensions[leaf] = 2
        return [[[leaf, np.array(h)]]]

    def relaxation_hamiltonian(self, coupling_list, prefix='R'):
        """
        Returns
        -------
        h_list: [[(str, ndarray)]]
        """
        omega_list, _, dim_list = [
            getattr(self, name) for name in self.INNER
        ]
        elec_leaf = self.elec_leaf
        projector = np.array([[0., 1.],
                              [1., 0.]])
        h_list = []
        zipped = zip(dim_list, omega_list, coupling_list)
        for n, (dim, omega, c) in enumerate(zipped):
            ph = Phonon(dim, omega)
            leaf = prefix + str(n)
            # e-ph part
            h_list.append([[leaf, omega * ph.coordinate_operator],
                           [elec_leaf, (c / omega) * projector]])
        return h_list

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
            self.leaves.append(leaf)
            self.dimensions[leaf] = dim
            # ph part
            h_list.append([[leaf, ph.hamiltonian]])
            # e-ph part
            h_list.append([[leaf, -omega * ph.coordinate_operator],
                           [elec_leaf, 2. * (c / omega) * projector]])
            # e part
            h_list.append([[elec_leaf, 2. * (c / omega) ** 2 * projector]])
        return h_list

    def inner_hamiltonian(self, prefix='I'):
        omega_list, lambda_list, dim_list = [getattr(self, name)
                                             for name in self.INNER]
        coupling_list = list(np.sqrt(np.array(lambda_list) / 2) *
                             np.array(omega_list))
        self.inner_prefix = prefix
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
        self.outer_prefix = prefix
        return self.vibration_hamiltonian(omega_list, coupling_list, dim_list,
                                          prefix=prefix)

    @property
    def inner_leaves(self):
        return [leaf for leaf in self.leaves
                if leaf.startswith(self.inner_prefix)]

    @property
    def outer_leaves(self):
        return [leaf for leaf in self.leaves
                if leaf.startswith(self.outer_prefix)]


class RotSpinBosonModel(SpinBosonModel):
    def __init__(self, including_bath=True, relaxation_list=None,
                 **kwargs):
        """ Needed parameters:
        - (for electronic part)
            'e1', 'e2', 'v',
        - (for inner part)
            'omega_list', 'lambda_list', 'dim_list',
        - (for outer part)
            'stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d'
        """
        super(RotSpinBosonModel, self).__init__(including_bath,
                                                relaxation_list,
                                                **kwargs)
        valid_attributes = self.ELEC + self.INNER + self.OUTER + self.FIELD
        for name, value in kwargs.items():
            if name in valid_attributes:
                setattr(self, name, value)
            else:
                logging.warning(__('Parameter {} unexpected, ignored.', name))
        self.elec_leaf = None
        self.inner_prefix = None
        self.outer_prefix = None
        self.leaves = []
        self.dimensions = {}
        self.including_bath = including_bath
        h_list = self.electron_hamiltonian()
        all_vibriations = (self.inner_hamiltonian() + self.outer_hamiltonian()
                           if including_bath else self.inner_hamiltonian())
        h_list.extend(all_vibriations)
        if relaxation_list is not None:
            h_list.extend(
                self.relaxation_hamiltonian(
                    coupling_list=relaxation_list, prefix='I')
            )
        self.h_list, self.f_list = self.collect_electric_terms(h_list)
        return


# EOF
