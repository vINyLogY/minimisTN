#!/usr/bin/env python2
# coding: utf-8

import logging
import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
