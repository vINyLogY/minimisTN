#!/usr/bin/env python
# coding: utf-8

from builtins import range, map, zip
from itertools import tee
from operator import itemgetter


def unzip(iterable):
    """Unzip is the same as zip(*iter) but returns iterators, instead
    of expand the iterator. Mostly used for large sequence.

    Reference: https://gist.github.com/andrix/1063340
    """

    _tmp, iterable = tee(iterable, 2)
    iters = tee(iterable, len(_tmp.next()))
    return (map(itemgetter(i), it) for i, it in enumerate(iters))
