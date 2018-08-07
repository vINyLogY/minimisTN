#!/usr/bin/env python
# coding: utf-8
"""Convenient objects about meta-programming and logging.
"""
from __future__ import absolute_import, division

from builtins import map, range, zip
from itertools import tee
import logging

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


class Message(object):
    def __init__(self, fmt, args):
        self.fmt = fmt
        self.args = args

    def __str__(self):
        return self.fmt.format(*self.args)


class StyleAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super(StyleAdapter, self).__init__(logger, extra or {})

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, Message(msg, args), (), **kwargs)


class LogLevel(object):
    ERROR = logging.INFO
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    DEBUG1 = logging.DEBUG - 1
    DEBUG2 = logging.DEBUG - 2
    DEBUG3 = logging.DEBUG - 3

logger = StyleAdapter(logging.getLogger(__name__))
