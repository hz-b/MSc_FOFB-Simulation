#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Bessy Signal module

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""

import matplotlib.plt as plt
import numpy as np

import mysignal as ms


def corrector_order1():
    A = 3
    a = A / (2*np.pi*3)
    return ms.TF([A], [a, 1])
