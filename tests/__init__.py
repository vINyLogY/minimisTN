#!/usr/bin/env python2
# coding: utf-8
import importlib
import logging
import os
import sys

__all__ = [
    'sine_dvr',
    'po_dvr',
    'dvr_with_sympy',
    'propagation',
    'heisenberg'
]

# Add minimisTN/ to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

# Root logger settings
level = logging.DEBUG if __debug__ else logging.INFO
logging.basicConfig(
    format='(In %(module)s)  %(message)s',
    stream=sys.stderr, level=level
)
