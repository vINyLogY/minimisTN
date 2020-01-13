#!/usr/bin/env python2
# coding: utf-8
import logging
import sys
# Basic logging setting
level = logging.DEBUG
logging.basicConfig(
    format='(In %(module)s) %(message)s',
    stream=sys.stdout, level=level
)
