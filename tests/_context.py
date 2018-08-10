#!/usr/bin/env python2
# coding: utf-8
"""Make each file in tests/ executable for debug convenience
"""

import logging
import sys
import os

# Add minimisTN/ to sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

# Root logger settings
level = logging.DEBUG if __debug__ else logging.INFO
logging.basicConfig(
    format='(In %(funcName)s, %(module)s)  %(message)s',
    stream=sys.stdout, level=level
)
