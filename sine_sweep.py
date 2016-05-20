#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""

from __future__ import division, print_function
from math import log, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mysignal as ms

plt.close('all')

fs = 150
Ts = 1/fs
T = 20
t = np.linspace(0, T, fs*T+1)
N = t.size

f1 = 0.01
f2 = fs/2
f_exp = f1*T/log(f2/f1)*(np.exp(log(f2/f1)*t/T)-1)
f_lin = 2*f1*t+(f2-f1)/T*0.5*(t**2)

x = np.sin(2*pi*f_lin)
plt.figure()
plt.plot(np.fft.fftfreq(N, Ts), abs(np.fft.fft(x)))
H = ms.TF([1, 2], [3, 1, 1])
H.plotHw(2*pi*np.logspace(-2, 2))

y = np.zeros(N)
for k in range(1, N):
    y[k] = H.apply_f(x[:k+1], y[:k], Ts)

plt.figure()
plt.subplot(411)
plt.plot(x)
plt.subplot(412)
plt.plot(y)

y += np.random.randn(N)*.0  # Add noise
plt.subplot(413)
plt.plot(y)

hammingw = np.hamming(2*N)  # Hamming window

y *= hammingw[N:]
plt.subplot(414)
plt.plot(y)


## WITH FFT
Y = np.fft.fft(y)
X = np.fft.fft(x)
H2 = Y / X

freq_plt = np.fft.fftfreq(N, Ts)[:int(N/2)]
H2 = H2[:int(N/2)]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freq_plt, abs(H2))
plt.title("Experimental")
plt.xscale('log')
plt.yscale('log')
plt.grid(which="both")
plt.subplot(2, 1, 2)
plt.plot(freq_plt, np.unwrap(np.angle(H2))[:N])
plt.xscale('log')
plt.grid(which="both")

## WITH AUTO/X-CORR
a = signal.correlate(x, x, "same")
c = signal.correlate(y, x, "same")

H2 = np.fft.fft(c) / np.fft.fft(a)

freq_plt = np.fft.fftfreq(N, Ts)[:int(N/3)]
H2 = H2[:int(N/3)]

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(freq_plt, abs(H2))
plt.title("Experimental")
plt.xscale('log')
plt.yscale('log')
plt.grid(which="both")
plt.subplot(2, 1, 2)
plt.plot(freq_plt, np.unwrap(np.angle(H2)))
plt.xscale('log')
plt.grid(which="both")
