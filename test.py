#! /usr/bin/env python3
# encoding: utf-8

from math import pi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mysignal as ms

plt.close('all')
fs = 150
tmax = 1
numtaps = 15
t = np.arange(int(tmax*fs))/fs
f = 10
T = pi
nyqf = fs/2

u = np.sin(2*pi*f*t+pi/4)

ntap = 15
h = np.cos(2*pi*f*np.arange(ntap)/fs+T-pi/4)*2/ntap

w, hw = signal.freqz(h)
y = signal.lfilter(h, [1], u)
plt.figure()
plt.plot(t, u)
plt.plot(t, y)

plt.figure()
plt.subplot(211)
plt.plot(w*nyqf/pi, abs(hw))
plt.subplot(212)
plt.plot(w*nyqf/pi, 180/pi*np.unwrap(np.angle(hw)))


for i in range(ntap, u.size):
    y[i] = np.flipud(u[i-ntap:i]).dot(h)

plt.figure()
plt.plot(t, u)
plt.plot(t, y)


N = 150
u = np.cos(2*pi*np.arange(N)/150) + np.random.random(N)

h = ms.TF(150,[1,5,150])

x = np.zeros((h.den.size-1,1))

(a, b, c, d, _) = signal.cont2discrete((h.A, h.B, h.C, h.D), 1/150,
                                       method='bilinear')
plt.legend()

#--- EASY PID ---#
fs = 150
#r = np.concatenate((np.zeros(10), [1], np.zeros(10)))
d = np.concatenate((np.zeros(10), [1], np.ones(10)))
pid = ms.PID(0, 0.8*fs, 0)
x = np.zeros(pid.den.size - 1)
N = d.size
t = np.arange(N)/fs
y = np.zeros(N)
yt = np.zeros(N)
e = np.zeros(N)
for k in range(1, N):
    e[k] = 0 - y[k-1]
#    yt[k] = pid.apply_f(e[:k+1], 1/fs)
    yt[k] = pid.apply_f(e[:k+1], 1/fs)
    y[k] = yt[k] + d[k]

plt.figure()
plt.plot(t, d)
plt.plot(t, y, '-v')

