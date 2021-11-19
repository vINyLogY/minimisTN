#!/usr/bin/env python
# coding: utf-8
"""Interface to logging package.
"""
import os
import sys
import logging

class Logger(object):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,
        filename=None, level='info',
        stream_fmt=None,
        file_fmt='%(message)s'
    ):
        if filename is None:
            # Use the same name of the main script as the default name
            filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.log'
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.levels[level])
        if stream_fmt is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter(stream_fmt))
            self.logger.addHandler(sh)  
        th = logging.FileHandler(filename=filename, mode='w', encoding='utf-8')
        th.setFormatter(logging.Formatter(file_fmt))
        self.logger.addHandler(th)

class BraceMessage:
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)
