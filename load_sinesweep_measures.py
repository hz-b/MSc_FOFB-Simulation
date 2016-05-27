#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""


from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import io as sio
import mysignal as ms

if __name__ == "__main__":
    l = 0.5
    plt.close('all')
    plt.ion()
    data = np.load('/home/churlaud/sine_sweep_2016-05-23_16-34-16.npy',
                   encoding='latin1')[0]
    u = data['input']
    xx = data['data']['xx'][:, 0, :]
    xy = data['data']['xy'][:, 0, :]
    yx = data['data']['yx'][:, 0, :]
    yy = data['data']['yy'][:, 0, :]

    N = u.size
    fs = 150
    freqs = np.fft.fftfreq(N, 1/fs)[:int(N/2)]

    plottitle = lambda cm, bpm: "1st {} corrector, all {} bpms shown".format(cm, bpm)
    H, f = ms.TF_from_signal(xx[3,:], u, fs, False, plottitle('x','x'))
#    ms.TF_from_signal(xy, u, fs, plottitle('x','y'))
#    ms.TF_from_signal(yx, u, fs, plottitle('y','x'))
#    ms.TF_from_signal(yy, u, fs, plottitle('y','y'))

    def tf(s, b0,b1,b2,a1,a2,a3,a4, a5,a6):
        b = b0 + b1*s + b2*s**2
        a =  1 + a1*s + a2*s**2 + a3*s**3  + a4*s**4+ a5*s**5 + a6*s**6
        return  b/a

    def tf_abs(s,b0,b1,b2, a1,a2,a3,a4,a5,a6):
        h = tf(s, b0,b1,b2, a1,a2,a3,a4,a5,a6)
        return abs(h)#+l*np.angle(h)

    s = 1j*2*np.pi*freqs
    fn = 80
    wn = 2*np.pi*50
    dc = 10
    z = 1

    h0 = H[0,:]
    sio.savemat('tf', {'h0':h0, 'f': freqs})
    [b0,b1,b2, a1,a2,a3,a4,a5,a6],_ = optimize.curve_fit(tf_abs, s, abs(h0),maxfev=50000)
    HH = tf(s, b0,b1,b2, a1,a2,a3,a4, a5,a6)*np.exp(1j*5*np.pi)

    d = dict()
    d['num'] = [b0,b1,b2]
    d['den'] = [1, a1,a2,a3,a4,a5,a6]
    d['phase'] = 5*np.pi
    d['use'] = "H(w) = sum(num[k]*jw**k) /sum(den[k]*jw**k)*exp(j phase)"

    np.save('ring_tf', [d])

    freqs2 = np.linspace(0,500)
    s2 = 1j*2*np.pi*freqs2
    y = tf(s2, b0,b1,b2, a1,a2,a3,a4, a5,a6)*np.exp(1j*5*np.pi)

#    np.save('ring_tf', y)

    plt.figure()
    plt.subplot(211)
    plt.plot(freqs, abs(h0), '+g')
    plt.plot(freqs2, np.abs(y), '-r')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='both')
    plt.subplot(212)
    plt.plot(freqs, np.unwrap(np.angle(h0)), '+g')
    plt.plot(freqs2, -np.unwrap(np.angle(y))+np.pi, '-r')
    plt.grid(which='both')
    plt.xscale('log')
