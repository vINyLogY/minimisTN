#!/usr/bin/env python2
# coding: utf-8
import importlib
import logging

from tests import __all__


for test in __all__:
    case = importlib.import_module('tests.' + test)
    logging.info('Testing: {}'.format(case.__name__))
    case.main()
