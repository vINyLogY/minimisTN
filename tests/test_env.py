#!/usr/bin/env python2
# coding: utf-8
from __future__ import absolute_import, division

import logging
import sys
from builtins import input
from math import sqrt
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from minitn.lib.tools import BraceMessage as __

def main():
    logging.root.setLevel(logging.DEBUG)
    print(__("{0}, {1}!", 'hello', 'world'))
    logging.info('INfo!')

main()
