#!/usr/bin/env python
# coding: utf-8
"""Interface to logging package.
"""
import os
import sys
import logging
from logging import handlers

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
        filename=None, level='info', when='D', backupCount=3,
        stream_fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        file_fmt='%(message)s'
    ):
        if filename is None:
            filename = os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.log'
        self.logger = logging.getLogger(filename)
        self.logger.setLevel(self.levels[level]) 
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(stream_fmt))
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backupCount, encoding='utf-8'
        )
        th.setFormatter(logging.Formatter(file_fmt))
        self.logger.addHandler(sh)  
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('all.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')

class BraceMessage:
    def __init__(self, fmt, *args, **kwargs):
        self.fmt = fmt
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.fmt.format(*self.args, **self.kwargs)


__ = BraceMessage

