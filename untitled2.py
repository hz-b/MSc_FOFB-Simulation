#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""

from __future__ import division, print_function
from math import log, pi
import numpy as np
import matplotlib.pyplot as plt
import mysignal as ms

f1 = 0.01
f2 = 20
fs = 150
Ts = 1/fs
T = 20
t = np.linspace(0, T, fs*T+1)
N = t.size

s = np.sin(2*pi*f1*T/log(f2/f1)*(np.exp(log(f2/f1)*t/T)-1))
plt.figure()
plt.plot(t, s)

H = ms.TF([1], [1, 2])
H.plotHw(2*pi*np.logspace(-1, 3))

y = np.zeros(N)
for k in range(1, N):
    y[k] = H.apply_f(s[:k+1], y[:k], Ts)
plt.figure()
plt.plot(abs(np.fft.fft(y))/abs(np.fft.fft(s)))