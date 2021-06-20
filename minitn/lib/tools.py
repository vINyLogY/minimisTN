#!/usr/bin/env python
# coding: utf-8
"""Convenient objects about meta-programming and logging.
"""
from __future__ import absolute_import, division

import contextlib
import logging
from builtins import map, range, zip
from functools import wraps, partial
from itertools import tee
from operator import itemgetter
from time import time

import matplotlib.pyplot as plt
from matplotlib import rc


def unzip(iterable):
    """The same as zip(*iter) but returns iterators, instead
    of expand the iterator. Mostly used for large sequence.

    Reference: https://gist.github.com/andrix/1063340

    Parameters
    ----------
    iterable : iterable

    Returns
    -------
    unzipped : (iterator)
    """

    _tmp, iterable = tee(iterable, 2)
    iters = tee(iterable, len(next(_tmp)))
    return (map(itemgetter(i), it) for i, it in enumerate(iters))


class BraceMessage:
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


__ = BraceMessage


def time_this(func):
    @wraps(func)
    def _time_this(*args, **kwargs):
        start = time()
        r = func(*args, **kwargs)
        end = time()
        logging.warning(
            __('{}.{} : {}', func.__module__, func.__name__, end - start)
        )
        return r

    return _time_this


empty = object()


class Parameters(object):
    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        return


def iter_visitor(start, r, method='DFS'):
    """Iterative visitor.

    Parameters
    ----------
    start : obj
        Initial object
    r : obj -> [obj]
        Relation function.
    method : {'DFS', 'BFS'}, optional
        'DFS': Depth first; 'BFS': Breadth first.
    """
    stack, visited = [start], set()
    while stack:
        if method == 'DFS':
            vertex = stack.pop()
        elif method == 'BFS':
            vertex, stack = stack[0], stack[1:]
        else:
            raise NotImplementedError()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(r(vertex) - visited)
            yield vertex


@contextlib.contextmanager
def figure(*args, **kwargs):
    rc('font', family='Times New Roman')
    rc('text', usetex=True)
    fig = plt.figure(*args, **kwargs)
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    yield fig
    plt.close(fig)


def huffman_tree(sources, importances=None, obj_new=None, n_branch=2):
    def string(x): return x[0]

    def key(x): return x[1]

    if importances is None:
        importances = [1] * len(sources)
    if obj_new is None:
        def counter(x=0):
            x += 1
            return x
        
        obj_new = counter

    sequence = list(zip(sources, importances))
    graph = {}
    while len(sequence) > 1:
        sequence.sort(key=key)
        try:
            branch, sequence = sequence[:n_branch], sequence[n_branch:]
        except:
            branch, sequence = sequence, []
        p = sum(map(key, branch))
        new = obj_new()
        graph[new] = list(map(string, branch))
        sequence.insert(0, (new, p))
    return graph, string(sequence[0])
