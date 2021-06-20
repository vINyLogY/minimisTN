#!/usr/bin/env python
# coding: utf-8
r"""ML-MCTDH Algorithms

References
----------
.. [1] arXiv:1603.03039v4
.. [2] My slides
"""
from __future__ import absolute_import, division, print_function

import logging
from builtins import filter, map, range, zip
from functools import partial
from itertools import count, combinations_with_replacement
from contextlib import contextmanager

import numpy as np
from scipy import linalg, integrate, sparse

from minitn.lib.tools import __
from minitn.lib.numerical import DavidsonAlgorithm
from minitn.tensor import Tensor, Leaf


class MultiLayer(object):
    r"""Structure of the wavefunction/state::

          ... ... ... 
        n_0|   |   |n_p-1
          2_0 ... 2_p-1
             \ | /
          m_0 \|/ m_p-1
               1
               |
              ...

    and corresponding Hamiltonian:
    * Option 1::

          n_0/   \n_p-1
           h_0   h_p-1
        n_0/ \   / \n_p-1
               + r
               |
           -- ... --

    Note that the order of contraction is essential.
    """
    # Coefficient settings...
    hbar = 1.
    regular_err = 1.e-12
    svd_err = None
    svd_rank = None
    pinv = True
    max_ode_steps = 1000
    cmf_steps = 1
    ode_method = 'RK45'
    ode_in_real = False
    snd_order = False
    ps_method = 'split-unite'

    @classmethod
    def settings(cls, **kwargs):
        """
        Parameters
        ----------
        hbar : float
            Default = 1.
        regular_err : float
            Default = 1.e-12
        svd_err : float
            Error allowed for SVD; default is None.
        pinv : bool
            Whether to use `scipy.linalg.pinv2` for inversion.
            Default is True.
        max_ode_steps : int 
            Maximal steps allowed in one ODE solver; default = 1000.
        cmf_steps : int
            Upper bound for CMF steps; default = 1
        ode_method : {'RK45', 'RK23', ...}
            Name of `OdeSolver` in `scipy.intergate`.
        snd_order : boot
            [Deprecated] Whether to use 2nd order method in projector splitting.
            Note that 2nd order method should be more accurate, but its
            complexity is :math:`2^d`, where `d` is the depth of tree.
            Default is False.
        ps_method : string
            Method of projector-splitting.
            `s` for one-site method and `u` for two-site method.
        """
        for name, value in kwargs.items():
            if not hasattr(cls, name):
                raise AttributeError('{} has no attr \'{}\'!'
                                     .format(cls, name))
            setattr(cls, name, value)
        return

    def __init__(self, root, h_list, f_list=None, use_str_name=False):
        """
        Parameters
        ----------
        root : Tensor
        h_list : [[(Leaf, array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.  This is time independent part of Hamiltonian.
        f_list : [[(Leaf, float  ->  array)]]
            h_list is a list of `term`, where `term` is a list of tuple like
            `(Leaf, array)`.  This is time dependent part of Hamiltonian.
        use_str_name : bool
            whether to use str to replace Leaf above. Default is False.
        """
        self.root = root
        self.h_list = h_list
        self.f_list = f_list

        # For propogation purpose
        self.time = None
        self.overall_norm = 1
        self.init_energy = None

        # Type check
        for term in h_list:
            for pair in term:
                if not isinstance(pair[0], Leaf) and not use_str_name:
                    raise TypeError('0-th ary in tuple must be of type Leaf!')
                if np.array(pair[1]).ndim != 2:
                    raise TypeError('1-th ary in tuple must be 2-D ndarray!')
        if f_list is not None:
            for term in f_list:
                for pair in term:
                    if not isinstance(pair[0], Leaf) and not use_str_name:
                        raise TypeError('0-th ary in tuple must be of type '
                                        'Leaf!')
                    if not callable(pair[1]):
                        raise TypeError('1-th ary in tuple must be callable!')
        if use_str_name:
            all_terms = h_list if f_list is None else h_list + f_list
            leaves_dict = {leaf.name: leaf for leaf in root.leaves()}
            for term in all_terms:
                for _ in range(len(term)):
                    fst, snd = term.pop(0)
                    term.append((leaves_dict[str(fst)], snd))

        # Some cached data
        self.inv_density = {}    # {Tensor: ndarray}
        self.env_ = {}    # {(int, Tensor, int): ndarray}
        return

    def print_hamiltonian(self, verbose=False):
        """
        Parameters
        verbose : bool
            Whether to print the dense array of each term.
            Default is `False`.
        """
        for n, term in enumerate(self.h_list):
            print('Term {:d}'.format(n))
            for leaf, array in term:
                msg = "- At Leaf {}: {}".format(leaf.name, array.shape)
                print(msg)
                if verbose:
                    print(array)
        return

    @staticmethod
    def triangular(n_list):
        """A Generator yields the natural number in a triangular order.
        """
        length = len(n_list)
        prod_list = [1]
        for n in n_list:
            prod_list.append(prod_list[-1] * n)
        prod_list = prod_list

        def key(case):
            return sum(n * i for n, i in zip(prod_list, case))
 
        combinations = {0: [[0] * length]}
        for m in range(prod_list[-1]):
            if m not in combinations:
                permutation = [
                    case[:j] + [case[j] + 1] + case[j + 1:]
                    for case in combinations[m - 1] for j in range(length)
                    if case[j] + 1 < n_list[j]
                ]
                combinations[m] = []
                for case in permutation:
                    if case not in combinations[m]:
                        combinations[m].append(case)
            for case in combinations[m]:
                yield key(case)

    def _local_matvec(self, leaf):
        h = None
        for term in self.h_list:
            if len(term) == 1 and term[0][0] is leaf:
                h = term[0][1]
                break
        if h is None and self.f_list is not None:
            for term in self.f_list:
                if len(term) == 1 and term[0][0] is leaf:
                    h = term[0][1]
                    h = h(0.0)
                    break
        if h is None:
            raise RuntimeError('Cannot found a local hamiltonian for {}'
                                .format(leaf))

        def matvec(vec, mat=h):
            vec = np.reshape(vec,  h.shape)
            ans = np.dot(mat, vec)
            return np.reshape(ans, -1)
        return matvec

    def autocomplete(self, n_bond_dict, max_entangled=False):
        """Autocomplete the tensors linked to `self.root` with suitable initial
        value.

        Parameters
        ----------
        n_bond_dict : {Leaf: int}
            A dictionary to specify the dimensions of each primary basis.
        max_entangled : bool
            Whether to use the max entangled state as initial value (for finite
            temperature and imaginary-time propagation).  Default is `False`.
        """
        for t in self.root.visitor(leaf=False):
            try:
                t.array
            except AttributeError:
                axis = t.axis
                if max_entangled and not any(t.children(leaf=False)):
                    if len(list(t.children(leaf=True))) != 2 or axis is None:
                        raise RuntimeError('Not correct tensor graph for FT.')
                    for i, leaf, j in t.children():
                        if not leaf.name.endswith("'"):
                            n_leaf = n_bond_dict[(t, i, leaf, j)]
                            break
                    p, p_i = t[axis]
                    n_parent = n_bond_dict[(p, p_i, t, axis)]
                    vec_i = np.diag(np.ones((n_leaf,)) / np.sqrt(n_leaf))
                    vec_i = np.reshape(vec_i, -1)
                    init_vecs = [vec_i]
                    da = DavidsonAlgorithm(self._local_matvec(leaf),
                                           init_vecs=init_vecs,
                                           n_vals=n_parent)
                    array = da.kernel(search_mode=True)
                    if len(array) >= n_parent:
                        array = array[:n_parent]
                    else:
                        for j in range(n_parent - len(array)):
                            v = np.zeros((n_leaf ** 2,))
                            v[j] = 1.0
                            array.append(v)
                    assert len(array) == n_parent
                    assert np.allclose(array[0], vec_i)
                    array = np.reshape(array, (n_parent, n_leaf, n_leaf))
                else:
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
                    for n, v_i in zip(self.triangular(n_children), array):
                        v_i[n] = 1.
                    array = np.reshape(array, shape)
                    if axis is not None:
                        array = np.moveaxis(array, 0, axis)
                t.set_array(array)
                t.normalize(forced=True)
                assert (t.axis is None or
                        np.linalg.matrix_rank(t.local_norm()) ==
                        t.shape[t.axis])
        if __debug__:
            for t in self.root.visitor():
                t.check_completness(strict=True)
        return

    def term_visitor(self, use_cache=False, op=None, only_td=False):
        """Visit all terms in self.h_list.

        Parameters
        ----------
        use_cache : bool
            Whether to load the cached mean fields in `self.env_` to 
            `tensor._partial_env`.  Default is `False`.
        op : [[(Leaf, array | float  ->  array)]]
            The operator to be visited term by term. 
            Default is `self.h_list` (and `self.f_list`, if any). 
        """
        time = self.time
        if time is None:
            time = 0.0
        visitor = self.root.visitor
        for tensor in visitor(axis=None):
            tensor.reset()
        if op is None:
            if self.f_list is not None:
                all_terms = self.h_list + self.f_list
            else:
                all_terms = self.h_list
        else:
            all_terms = op
        for n, term in enumerate(all_terms):
            if op is None and only_td and n < len(self.h_list):
                continue
            for tensor in visitor(axis=None, leaf=False):
                tensor.reset()
                if use_cache:
                    for i, t, j in tensor.linkages:
                        if (n, t, j) in self.env_:
                            tensor.load_cache(i, self.env_[(n, t, j)])
            for leaf, array in term:
                if callable(array):
                    array = array(time)
                if isinstance(leaf, Leaf):
                    leaf.set_array(array)
                else:
                    raise RuntimeError('{} is not leaf in term {}'
                                       .format(leaf, term))
            yield n
            for leaf, _ in term:
                leaf.reset()

    def matrix_element(self, op=None):
        """Return the matrix element with the states of the network which
        `self.root` in.  Sum over `self.h_list`.
        """
        ans = 0.0
        for _ in self.term_visitor(op=op):
            ans += self.root.matrix_element()
        return ans

    def expection(self, normalized=False, op=None):
        """Return the expection value with the state of the network which
        `self.root` in.  Sum over `self.h_list`.
        """
        ans = 0.0
        for _ in self.term_visitor(op=op):
            ans += self.root.expection()
        if normalized:
            ans /= self.root.global_square()
        np.testing.assert_almost_equal(np.imag(ans), 0) 
        return np.real(ans)

    # @profile
    def _single_eom(self, tensor, n, cache=False):
        """C.f. `Multi-Configuration Time Dependent Hartree Theory: a Tensor
        Network Perspective`, p38. This method does not contain the `i hbar`
        coefficient.

        Parameters
        ----------
        tensor : Tensor
            Must in a graph with all nodes' array set, including the leaves.
        n : int
            No. of Hamiltonian term.

        Return:
        -------
        array : ndarray
            With the same shape with tensor.shape.
        """
        partial_product = Tensor.partial_product
        partial_trace = Tensor.partial_trace
        partial_env = tensor.partial_env

        # Env Hamiltonians
        tmp = tensor.array
        for i in range(tensor.order):
            try:
                env_ = self.env_[(n, tensor, i)]
            except KeyError:
                env_ = partial_env(i, proper=True)
                if cache:
                    self.env_[(n, tensor, i)] = env_
            tmp = partial_product(tmp, i, env_)
        # For non-root nodes...
        if tensor.axis is not None:
            # Inversion
            axis, inv = tensor.axis, self.inv_density[tensor]
            tmp = partial_product(tmp, axis, inv)
            # Projection
            tmp_1 = np.array(tmp)
            array = tensor.array
            conj_array = np.conj(array)
            tmp = partial_trace(tmp, axis, conj_array, axis)
            tmp = partial_product(array, axis, tmp, j=1)
            tmp = (tmp_1 - tmp)
        return tmp

    def _form_inv_density(self):
        self.inv_density = {}
        visitor = self.root.visitor
        for tensor in visitor(axis=None):
            tensor.reset()
        for tensor in visitor(axis=None):
            axis = tensor.axis
            if axis is not None:
                density = tensor.partial_env(axis, proper=True)
                if self.pinv:
                    inv = linalg.pinv2(density)
                else:
                    inv = linalg.inv(density + self.regular_err *
                                     np.identity(tensor.shape[axis]))
                self.inv_density[tensor] = inv
        return self.inv_density

    # @profile
    def _form_env(self, update=False):
        self.env_ = {}
        network = self.root.visitor(axis=None, leaf=False)
        for n in self.term_visitor(only_td=update):
            for tensor in network:
                for i in range(tensor.order):
                    env_ = tensor.partial_env(i, proper=True)
                    self.env_[(n, tensor, i)] = env_
        return self.env_

    def dense_hamiltonian(self):
        h_list = self.h_list
        f_list = self.f_list
        if f_list is not None:
            raise NotImplementedError()
        leaves = self.root.leaves()
        dims = {}
        for l in leaves:
            t, i = l[0]
            dims[l] = t.shape[i]
        ans = 0.
        for term in h_list:
            h_i = 1
            term_ = dict(term)
            for l in leaves:
                h_ij = term_.get(l, np.identity(dims[l]))
                h_i = np.kron(h_i, h_ij)
            ans = ans + h_i
        return ans

    def eom(self, check=False):
        r"""Write the derivative of each Tensor in tensor.aux.

                   .
            g ::= <t|t> = 0

        Parameters
        ----------
        check : bool
            True to check the linkage completness.
        """
        visitor = self.root.visitor
        if check:
            for tensor in visitor():
                tensor.check_completness(strict=True)
        # Clean
        for t in visitor():
            t.aux = None
        # Term by term...
        for n in self.term_visitor():
            for tensor in visitor(leaf=False):
                tmp = self._single_eom(tensor, n)
                prev = tensor.aux
                tensor.aux = tmp if prev is None else prev + tmp
        # Times coefficient
        for tensor in visitor(leaf=False):
            tensor.aux /= self.coefficient()
            if tensor.axis is None and self.init_energy is not None:
                tensor.aux -= self.init_energy * tensor.array
        return

    def coefficient(self):
        imaginary = self.time is None
        return -self.hbar if imaginary else 1.0j * self.hbar

    def direct_step(self, ode_inter=0.01, imaginary=False):
        visitor = self.root.visitor
        self._form_env()
        self._form_inv_density()
        method = self.ode_method
        if method == 'Newton':
            self.eom()
            for t in visitor(leaf=False):
                y0 = t.array
                dy = ode_inter * t.aux
                t.set_array(y0 + dy)
                t.aux = None
        elif method == 'RK4':
            k = [{}, {}, {}, {}]  # save [y0, k1, k2, k3]
            eom = self.eom
            eom()    # for k1
            for t in visitor(leaf=False):
                y0 = t.array
                k1 = ode_inter * t.aux
                t.set_array(y0 + k1 / 2)
                k[0][t] = y0
                k[1][t] = k1
            eom()    # for k2
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k2 = ode_inter * t.aux
                t.set_array(y0 + k2 / 2)
                k[2][t] = k2
            eom()    # for k3
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k3 = ode_inter * t.aux
                t.set_array(y0 + k3)
                k[3][t] = k3
            eom()    # for k4
            for t in visitor(leaf=False):
                y0 = k[0][t]
                k4 = ode_inter * t.aux
                t.set_array(
                    k[0][t] +
                    (k[1][t] + 2. * k[2][t] + 2. * k[3][t] + k4) / 6.
                )
                t.aux = None
        else:
            root = self.root

            def diff(t, y):
                """This function will not change the arrays in tensor network.
                """
                origin = root.vectorize()
                root.tensorize(y)
                self.time = t if not imaginary else None
                self.eom()
                ans = root.vectorize(use_aux=True)
                root.tensorize(origin)
                return ans

            def reformer():
                self._form_env()
                self._form_inv_density()
                return

            def updater(y):
                root.tensorize(y)
                for t in root.visitor(leaf=False):
                    t.normalize()

            y0 = root.vectorize()
            self._solve_ode(diff, y0, ode_inter, reformer, updater)
        return

    @contextmanager
    def log_inner_product(self, level=logging.DEBUG):
        root = self.root
        if logging.root.isEnabledFor(level):
            shape_dict = {}
            init = root.vectorize(shape_dict=shape_dict)
        try:
            yield self
        except:
            pass
        else:
            if logging.root.isEnabledFor(level):
                root.tensorize(np.conj(init), use_aux=True,
                               shape_dict=shape_dict)
                ip = root.global_inner_product()
                logging.log(level, __("<|>:{}", ip))

    def _solve_ode(self, diff, y0, ode_inter, reformer, updater):
        OdeSolver = getattr(integrate, self.ode_method)
        in_real = self.ode_in_real
        t0 = self.time
        if t0 is None:
            t0 = 0.0
        t1 = t0 + ode_inter
        if in_real:
            y0 = np.array((y0.real, y0.imag), dtype='float64')
            complex_diff = diff
            def diff(t, x):
                xc = np.array(x[0] + 1.0j * x[1], dtype='complex128')
                yc = complex_diff(t, xc)
                y = np.array((yc.real, yc.imag), dtype='float64')
                return y
                
        ode_solver = OdeSolver(diff, t0, y0, t1, vectorized=False)
        cmf_steps = self.cmf_steps
        for n in count(1):
            if ode_solver.status != 'running':
                logging.debug(__('* Propagation done.  Average CMF steps: {}',
                                n // cmf_steps))
                break
            if n % cmf_steps == 0:
                if n >= self.max_ode_steps:
                    msg = __('Reach ODE limit {}', n)
                    logging.warning(msg)
                    raise RuntimeWarning(msg)
                if reformer is not None:
                    reformer()
            ode_solver.step()
            updater(ode_solver.y)
        return

    def _split_prop(self, tensor, tau=0.01, imaginary=False, cache=False):
        def diff(t, y):
            """This function will not change the arrays in tensor network.
            """
            origin = tensor.array
            tensor.set_array(np.reshape(y, tensor.shape))
            ans = np.zeros_like(y)
            self.time = t if not imaginary else None
            for n in self.term_visitor(use_cache=True):
                ans += np.reshape(self._single_eom(tensor, n, cache=cache), -1)
            if self.init_energy is not None:
                ans -= self.init_energy * y
            ans /= self.coefficient()
            tensor.set_array(origin)
            return ans

        def reformer():
            self._form_env(update=True)
            return

        def updater(y):
            tensor.set_array(np.reshape(y, tensor.shape))
            tensor.normalize(forced=(not imaginary))

        if tensor.axis is None:
            self.root = tensor
        else:
            raise RuntimeError("Cannot propagate on Tensor {}"
                               "which is not a root node!".format(tensor))
        logging.debug(__("* Propagating at {} ({}) for {}",
                        tensor, tensor.shape, tau))
        y0 = np.reshape(tensor.array, -1)
        _re = None if self.f_list is None else reformer
        with self.log_inner_product(level=logging.DEBUG):
            self._solve_ode(diff, y0, tau, _re, updater)
        return tensor

    def remove_env(self, *args):
        for n, _ in enumerate(self.h_list):
            for tensor in args:
                for i, _, _ in tensor.linkages:
                    if (n, tensor, i) in self.env_:
                        del self.env_[(n, tensor, i)]
        return

    def move(self, t, i, op=None, unite_first=False):
        end, _ = t[i]
        self.remove_env(t, end)
        if unite_first:
            path = t.unite_split(i, operator=op, rank=self.svd_rank,
                                 err=self.svd_err)
        else:
            path = t.split_unite(i, operator=op)
        self.root = end
        self.remove_env(*path)
        return end

    @staticmethod
    def snd_order_step(step, ode_inter=0.01, imaginary=False):
        half = 0.5 * ode_inter
        step(ode_inter=half, imaginary=imaginary, backward=False)
        step(ode_inter=half, imaginary=imaginary, backward=True)
        return 

    def split_step(self, ode_inter=0.01, imaginary=False, backward=False):
        self._form_env()
        prop = partial(self._split_prop, tau=ode_inter, imaginary=imaginary,
                       cache=True)
        inv_prop = partial(self._split_prop, tau=(-ode_inter),
                           imaginary=imaginary, cache=True)
        linkages = list(self.root.decorated_linkage_visitor(leaf=False))
        move = self.move
        if backward:
            prop(self.root)
            for rev, (t1, _, t2, j) in reversed(linkages):
                if rev:
                    move(t2, j, op=inv_prop)
                    prop(t1)
                else:
                    move(t2, j)
        else:
            for rev, (t1, i, _, _) in linkages:
                if not rev:
                    move(t1, i)
                else:
                    prop(t1)
                    move(t1, i, op=inv_prop)
            prop(self.root)
        return

    def unite_step(self, ode_inter=0.01, imaginary=False, backward=False):
        self._form_env()
        prop = partial(self._split_prop, tau=ode_inter, imaginary=imaginary,
                       cache=True)
        inv_prop = partial(self._split_prop, tau=(-ode_inter),
                           imaginary=imaginary, cache=True)
        linkages = list(self.root.decorated_linkage_visitor(leaf=False))
        move = self.move
        origin = self.root
        counter = {}
        for t in self.root.visitor(leaf=False):
            counter[t] = len(list(t.children(leaf=False)))
        if backward:
            n_origin = counter[origin]
            for rev, (t2, _, t1, i) in reversed(linkages):
                if rev:
                    if t2 is not origin or counter[t2] != n_origin:
                        inv_prop(t2)
                        counter[t2] -= 1
                    move(t1, i, op=prop, unite_first=True)
                else:
                    if counter[t2] > 0:
                        move(t1, i)
        else:
            for rev, (t1, i, t2, _) in linkages:
                if not rev:
                    if counter[t2] > 0:
                        move(t1, i)
                else:
                    move(t1, i, op=prop, unite_first=True)
                    counter[t2] -= 1
                    if t2 is not origin or counter[t2] > 0:
                        inv_prop(t2)
        assert not any(counter.values())
        return

    def r_split_step(self, ode_inter=0.01, imaginary=False,
                    _root=None, _axis=None):
        """Recursive projector-splitting method.  The time of the
        coefficient of a wfn matters most.
        """
        if _root is None:
            self._form_env()
            _root = self.root
        propagate = partial(self._split_prop, imaginary=imaginary)
        move = self.move

        def branch_prop(r, axis, tau, backward=False):
            op1, op2 = None, partial(propagate, tau=(-tau))
            linkages = list(r.children(axis=axis, leaf=False))
            if backward:
                op1, op2,  = op2, op1
            for i, t, j in linkages:
                move(r, i, op1)
                self.r_split_step(ode_inter=tau, imaginary=imaginary,
                                 _root=t, _axis=j)
                move(t, j, op2)
            return

        if self.snd_order:
            branch_prop(_root, _axis, 0.5 * ode_inter)
            propagate(_root, tau=ode_inter, cache=True)
            branch_prop(_root, _axis, 0.5 * ode_inter, backward=True)
        else:
            branch_prop(_root, _axis, ode_inter)
            propagate(_root, tau=ode_inter, cache=True)
        return

    def propagator(self, steps=None, ode_inter=0.01, split=False,
                   imaginary=False, start=0, move_energy=False):
        """Propagator generator

        Parameters
        ----------
        steps : int
        ode_inter : float
        method : {'Newton', 'RK4', 'RK45', ...}
        """
        if split:
            identifier = self.ps_method.upper()
            if identifier.startswith('U'):
                step = self.unite_step
            elif identifier.startswith('S'):
                step = self.split_step
            elif identifier.startswith('R'):
                step = self.r_split_step
            else:
                raise ValueError("No PS method '{}'!".format(self.ps_method))
            if self.snd_order and not identifier.startswith('R'):
                step = partial(self.snd_order_step, step)
        else:
            step = self.direct_step
        root = self.root
        expection = self.expection
        if move_energy:
            self.init_energy = expection(normalized=True)

        for n in count():
            if steps is not None and n >= steps:
                break
            time = start + n * ode_inter
            logging.info(__(
                "Propagating at t: {:.3f}, E: {:.8f}, |v|^2: {:.8f}",
                time,
                expection(normalized=True),
                root.global_square()
            ))
            self.time = time if not imaginary else None
            yield (time, root)
            try:
                step(ode_inter=ode_inter, imaginary=imaginary)
                if imaginary:
                    self.overall_norm *= self.root.normalize(forced=True)
            except RuntimeWarning:
                break

    def autocorr(self, steps=None, ode_inter=0.01, split=False,
                 imaginary=False, fast=False, start=0, move_energy=False):
        if not fast:
            _init = {}
            for t in self.root.visitor(leaf=False):
                _init[t] = t.array
        for time, r in self.propagator(steps=steps, ode_inter=ode_inter,
                                       split=split, imaginary=imaginary,
                                       start=start, move_energy=move_energy):
            for t in r.visitor(leaf=False):
                t.aux = t.array if fast else np.conj(_init[t])
            auto = r.global_inner_product()
            ans = (2. * time, auto) if fast else (time, auto)
            yield ans
            for t in r.visitor(leaf=False):
                t.aux = None

    @property
    def relative_partition_function(self):
        """Return Z(beta) / Z(0).
        """
        return self.overall_norm ** 2


if __name__ == '__main__':
    for n, i in enumerate(MultiLayer.triangular([10, 10, 10])):
        print('{:03d}'.format(i))
        if n > 30:
            break

# EOF
