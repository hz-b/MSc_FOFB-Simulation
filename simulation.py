#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi
import matplotlib.pyplot as plt
import numpy as np

import mysignal as ms
import search_kicks.tools as sktools

if __name__ == "__main__":

    plt.ion()
    plt.close('all')
    fs = 150.
    Ts = 1/fs
    t_max = 60

    t = np.linspace(0, t_max-Ts, fs*t_max)

    H_dip = ms.bessy.corrector_order1()

    wc = 2*pi*80
    Sx = np.load('SmatX.npy')
    w = np.std(Sx, 0)
    Sx = Sx / w
#    Sx = Sx[5:, 5:]
#    Sx = Sx[0, 0].reshape((1,1))
#    Sx = Sx[:, 0].reshape((Sx.shape[0],1))
    H_lp = ms.TF([1], [1/wc**2, 1.4142/wc, 1])  # Low pass butterworth

    sA, sB, sC, sD = np.load('ss_param.npy')
    H_ring = ms.TF(sA, sB, sC, sD)
    H_ring.num = H_ring.num.real
    H_ring.num *= H_ring.den[-1]/H_ring.num[-1]
    H_ring.num[0] = 0
#    pid = ms.PID(0.6, 1.0*fs, 0.3/fs)
    pid = ms.PID(0, 0.8*fs, 0./fs)

    delay_calc = 1e-3  # 2e-3
    delay_adc = 1e-4
    delay_dac = 3e-4
    delay = delay_calc + delay_adc + delay_dac

#    S_coeff =
#    H = S_coeff*H_lp*H_dip*pid
#    H = ms.PID(0, -0.8, 0)
#    G = H_ring /(1 - H*H_ring)
#    G.plotHw(w=np.logspace(-1, 2)*2*np.pi, bode=True)

    amplitude = 0.02
    perturbation = 'sinesweep'
    if perturbation == 'step':
        d = ms.inputs.step(t, 0.1, amplitude)
    elif perturbation == 'sinesweep':
        d = ms.inputs.sinesweep(t, fmin=1e-3, fmax=75, amplitude=amplitude)
    elif perturbation == 'sine':
        fsin = 10
        d = amplitude*np.sin(2*np.pi*t*fsin)
    elif perturbation == "impulse":
        d = ms.inputs.impulse(t, 0.1, amplitude)

#    H_ring = ms.TF([1],[1])
#    H_dip.num *= H_dip.den[-1]/H_dip.num[-1]
    H_dip = ms.TF([1], [1])
    delay = 0
    H_ring.plotStep()
#    H_dip.plotStep()
    H_ring.plotHw(np.logspace(-1,10))

#    H_dip.plotHw()

    y, x, fs_r = ms.bessy.simulate(d, pid, Sx, H_lp, H_dip, H_ring, delay, fs, True)

    ms.TF_from_signal(y, x, fs_r, method='fft', plot=True, plottitle='with delay')

