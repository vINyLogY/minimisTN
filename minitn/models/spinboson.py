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
    FIELD = ['mu', 'tau', 't_d', 'omega']

    def __init__(self, **kwargs):
        """ Needed parameters:
        - (for electronic part)
            'e1', 'e2', 'v',
        - (for inner part)
            'omega_list', 'lambda_list', 'dim_list',
        - (for outer part)
            'stop', 'n', 'dim', 'lambda_g', 'omega_g', 'lambda_d', 'omega_d'
        """
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
        h_list = self.electron_hamiltonian()
        h_list.extend(self.inner_hamiltonian() + self.outer_hamiltonian())
        self.h_list, self.f_list = self.collect_electric_terms(h_list)
        return

    def collect_electric_terms(self, h_list):
        def condition(term): return len(term) == 1 and term[0][0] == elec_leaf
        elec_list = filter(condition, h_list)
        elec_leaf = self.elec_leaf
        elec_array = sum([term[0][1] for term in elec_list])
        left_list = list(filterfalse(condition, h_list))
        try:
            field = self.td_electron_hamiltionian(elec_array)
            h_list = left_list
            f_list = [[[elec_leaf, field]]]
        except AttributeError:
            h_list = [[[elec_leaf, elec_array]]] + left_list
            f_list = None
        return h_list, f_list

    def td_electron_hamiltionian(self, ti_array):
        def field(t):
            mu, tau, t_d, omega = [getattr(self, name) for name in self.FIELD]
            h = [[0., mu],
                 [mu, 0.]]
            delta = t - t_d
            coeff = (np.exp(-4. * np.log(2. * delta ** 2 / tau ** 2)) *
                     np.cos(omega * delta))
            return coeff * np.array(h) + ti_array
        return field

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
        self.dimensions[leaf] = 2
        return [[[leaf, np.array(h)]]]

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


def huffman_tree(sources, importances=None, prefix='', n_branch=2):
    def string(x): return x[0]
    def key(x): return x[1]
    if importances is None:
        importances = [1] * len(sources)
    sequence = list(zip(sources, importances))
    graph = {}
    counter = 0
    while len(sequence) >= n_branch:
        sequence.sort(key=key)
        branch, sequence = sequence[:n_branch], sequence[n_branch:]
        p = sum(map(key, branch))
        new = prefix + '{:02d}'.format(counter)
        graph[new] = list(map(string, branch))
        sequence.insert(0, (new, p))
        counter += 1
    if n_branch > 2:
        graph[prefix + '{:02d}'.format(counter)] = list(map(string, sequence))
        counter += 1
    return graph, (prefix + '{:02d}'.format(counter - 1))


if __name__ == '__main__':
    from minitn.lib.units import Quantity
    from minitn.tensor import Leaf, Tensor
    from minitn.ml import MultiLayer
    logging.basicConfig(
        format='(In %(module)s)[%(funcName)s] %(message)s',
        level=logging.INFO
    )
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
        stop=Quantity(3 * 2250, 'cm-1').value_in_au,
        n=16,
        dim=30,
        lambda_g=Quantity(2250, 'cm-1').value_in_au,
        omega_g=Quantity(500, 'cm-1').value_in_au,
        lambda_d=Quantity(1250, 'cm-1').value_in_au,
        omega_d=Quantity(50, 'cm-1').value_in_au,
        mu=Quantity(250, 'cm-1').value_in_au,
        tau=Quantity(30, 'fs').value_in_au,
        t_d=Quantity(60, 'fs').value_in_au,
        omega=Quantity(13000, 'cm-1').value_in_au,
    )
    graph = {
        'ROOT': [sbm.elec_leaf, 'INNER', 'OUTER'],
        'INNER': ['L2-I0', 'L2-I1'],
        'L2-I0': sbm.inner_leaves[:2],
        'L2-I1': sbm.inner_leaves[2:],
    }
    outer_part, root = huffman_tree(sbm.outer_leaves, prefix='AUTO')
    outer_part['OUTER'] = outer_part.pop(root)
    graph.update(outer_part)
    root_node = Tensor.generate(graph, 'ROOT')
    root_node.is_normalized = True
    solver = MultiLayer(root_node, sbm.h_list, f_list=sbm.f_list,
                        use_str_name=True)
    bond_dict = {}
    for s, i, t, j in root_node.linkage_visitor():
        if isinstance(t, Leaf):
            bond_dict[(s, i, t, j)] = sbm.dimensions[t.name]
        else:
            bond_dict[(s, i, t, j)] = 10
    solver.autocomplete(bond_dict)
    solver.settings(
        cmf_steps=10,
        ode_method='RK23',
        ps_method='split-unite'
    )
    projector = np.array([[0., 0.],
                          [0., 1.]])
    op = [[[root_node[0][0], projector]]]
    li = []
    for time, _ in solver.propagator(
        steps=100,
        ode_inter=1.,
        split=False
    ):
        li.append((Quantity(time).convert_to(unit='fs'),
                   solver.expection(op=op)))
        msg = "Time: {}, P2: {}".format(*li[-1])
        print(msg)
    np.save('spin-boson', np.array(li))

# EOF
