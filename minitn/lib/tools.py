#!/usr/bin/env python
# coding: utf-8
"""Convenient functions about list and iterators.
"""
from __future__ import absolute_import, division

from builtins import map, range, zip
from itertools import tee
from operator import itemgetter


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
    iters = tee(iterable, len(_tmp.next()))
    return (map(itemgetter(i), it) for i, it in enumerate(iters))
