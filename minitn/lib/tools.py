#!/usr/bin/env python
# coding: utf-8
"""Convenient objects about meta-programming and logging.
"""
from __future__ import absolute_import, division

import contextlib
import logging
from builtins import map, range, zip
from functools import wraps
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
    def timed(*args, **kwargs):
        start = time()
        r = func(*args, **kwargs)
        end = time()
        logging.debug(
            __('{}.{} : {}', func.__module__, func.__name__, end - start)
        )
        return r
    return timed


@contextlib.contextmanager
def figure(*args, **kwargs):
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    rc('text', usetex=True)
    fig = plt.figure(*args, **kwargs)
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    yield fig
    plt.close(fig)
