#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""

from __future__ import division, print_function, unicode_literals

import copy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import search_kicks.tools as sktools

plt.close('all')

N = 2000
Fs = 150
t = np.arange(N)/Fs

freqs = np.fft.fftfreq(N, 1/Fs)
freqs_half = freqs[:N//2+1]
cm_fft = 5*np.random.random(N//2+1)*np.exp(1j*2*np.pi*np.random.random(N//2+1))

idxmin = np.argmin(abs(freqs_half - 9))
idx20 = np.argmin(abs(freqs_half - 20))
for k in range(idxmin, idx20):
    cm_fft[k] = 0.1*cm_fft[k]*(5 - (freqs_half[k] - 11)*(freqs_half[k] - 20))

nprand = np.random.random
cmph10 = 2*np.pi*nprand()
cm_fft[np.argmin(abs(freqs_half - 0))] = 0
cm_fft[np.argmin(abs(freqs_half - 10))] = 20*np.exp(1j*cmph10)
cm_fft[np.argmin(abs(freqs_half - 50))] = 30*np.exp(1j*2*np.pi*nprand())
cm_fft[-1] = 0

cm_fft = np.concatenate((cm_fft[:-1], np.flipud(cm_fft.conjugate())[:-1]))
cm_fft *= 500*1e-6/np.max(cm_fft)*N/2
cm = np.fft.ifft(cm_fft).real
#
#plt.figure()
#plt.plot(freqs, abs(cm_fft))
#plt.plot(freqs, abs(np.fft.fft(cm)) *2/N, '--r')

Sxx = np.load('SmatX.npy')
Syy = np.load('SmatY.npy')
Sxx_inv = sktools.maths.inverse_with_svd(Sxx, 48)
Syy_inv = sktools.maths.inverse_with_svd(Syy, 36)

BPMx = np.zeros((104, N))
BPMx += cm
BPMy = np.zeros((104, N))
CMx = Sxx_inv.dot(BPMx)
CMy = Syy_inv.dot(BPMy)
#CMx = np.zeros((Sxx.shape[1], N))
#CMy = np.zeros((Syy.shape[1], N))

#CMx[0, :] = 1e-4*np.cos(2*np.pi*10*np.arange(N)/Fs + 1.1)
#BPMx = Sxx.dot(CMx)
#BPMy = Syy.dot(CMy)
plt.figure()
plt.plot(abs(np.fft.fft(CMx[0,:])))

o = sktools.io.OrbitData(BPMx=BPMx,
                         BPMy=BPMy,
                         CMx=CMx,
                         CMy=CMy,
                         sampling_frequency=150.)

SAMPLE_NB = o.sample_number
#o.BPMx = o.BPMx[:-1,:]
#o.BPMy = o.BPMy[:-1,:]


#acosX, asinX = sktools.maths.extract_sin_cos(o.BPMx, fs=150., f=10.)
#acosY, asinY  = sktools.maths.extract_sin_cos(o.BPMy, fs=150., f=10.)
#valuesX = acosX - 1j*asinX
#valuesY = acosY - 1j*asinY

aX, pX = sktools.maths.extract_sin_cos(o.BPMx, fs=150., f=10., output_format='polar')
aY, pY = sktools.maths.extract_sin_cos(o.BPMy, fs=150., f=10., output_format='polar')

valuesX = aX*np.exp(1j*pX)
valuesY = aY*np.exp(1j*pY)
CorrX = np.dot(Sxx_inv, valuesX)
CorrY = np.dot(Syy_inv, valuesY)
ampX = np.abs(CorrX)
phX = np.angle(CorrX)

ampY = np.abs(CorrY)
phY = np.angle(CorrY)

t = np.arange(o.sample_number)/o.sampling_frequency
sin10 = 1e5*np.cos(2*np.pi*10*t+1.1)
amp10, ph10 = sktools.maths.extract_sin_cos(sin10.reshape(1, SAMPLE_NB),
                                            150., 10., 'polar')

#plt.plot(t, sin10, '+')
#plt.plot(t, amp10*np.cos(2*np.pi*10*t+ph10))
#plt.xlim((0, .3))
ybuf = np.zeros(15)
tx = np.arange(15).reshape(1,15).repeat(CorrX.size, axis=0)/150.
ty = np.arange(15).reshape(1,15).repeat(CorrY.size, axis=0)/150.
px = phX.reshape((phX.size,1)).repeat(15, axis=1) - ph10
py = phY.reshape((phY.size,1)).repeat(15, axis=1) - ph10
hx = np.cos(2*np.pi*10*tx - px)*2/15
hy = np.cos(2*np.pi*10*ty - py)*2/15

ex = np.zeros((CorrX.size, SAMPLE_NB))
ey = np.zeros((CorrY.size, SAMPLE_NB))
for k in range(SAMPLE_NB):
    for i in range(ybuf.size-1):
        ybuf[i] = ybuf[i+1]
    ybuf[-1] = sin10[k]
    for i in range(CorrX.size):
        ex[i, k] = ampX[i]/amp10*hx[i,:].dot(np.flipud(ybuf))
    for i in range(CorrY.size):
        ey[i, k] = ampY[i]/amp10*hy[i,:].dot(np.flipud(ybuf))


b = copy.deepcopy(o)
print(max(b.CMx[0,:]))
print(max(b.CMy[0,:]))

print(max(ex[0,:]))
b.CMx -= ex
b.CMy -= ey

for k in range(SAMPLE_NB):
    b.BPMx[:, k] = np.dot(Sxx, b.CMx[:,k])
    b.BPMy[:, k] = np.dot(Syy, b.CMy[:,k])
print(max(b.CMx[0,:]))
print(max(b.CMy[0,:]))
print(max(b.BPMx[0,:]))
print(max(b.BPMy[0,:]))
plt.figure('origin')
o.plot_fft(0)
plt.figure('dynam')
b.plot_fft(0)
plt.show()
