#!/usr/bin/env python2
# coding: utf-8
from __future__ import absolute_import, division

import logging
import sys
from builtins import input
from functools import partial
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

from minitn.dvr import SineDVR
from minitn.lib.numerical import PotentialFunction, WindowFunction, expection
from minitn.lib.tools import BraceMessage as __
from minitn.lib.tools import figure, time_this
from minitn.point import Point


def linear(x, c=0.25):
    return sqrt(c) * x


@time_this
def ml(x0, L, m, n, v_func, c):
    spf = SineDVR(x0, x0 + L, n)



def main():
    import time
    x0, L, n = -5., 10., 40
    v_func = PotentialFunction.sho()
    for m in range(1, 11):
        ml(x0, L, m, n, v_func, c=0.25)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    main()
