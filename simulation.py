#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import pi
import time
import matplotlib.pyplot as plt
import numpy as np
import control
from scipy.io import loadmat
import seaborn as sns
sns.set_style('ticks')

import sys
sys.path.insert(0, '../search_kicks')
import mysignal as ms

def zefunc(K, fb):

    st= time.time()
    y, x, fs_r = ms.bessy.simulate(d, K, Sx, H_lp, H_dip, H_ring, delay, fs, False)
    ffty = np.abs(np.fft.fft(y[0,:]))[:N//2]*2/N
    ffty /= ffty[id50]
    plt.plot(freqs,ffty, label=r"$f_b = {}$ Hz".format(fb))

    print("{} s: needs {} s".format(t_max, time.time()-st))


if __name__ == "__main__":

    plt.close('all')

    corr_data = loadmat('data/correctors')

    correctors = corr_data['correctors'][0,:]


    fs = 150.
    Ts = 1/fs
    t_max = 3
    t = np.arange(int(fs*t_max))/fs
    N = int(fs*t_max)
    H_dip = ms.bessy.corrector_order1()

    wc = 2*pi*80
    Sx = np.load('data/SmatX.npy')
    w = np.std(Sx, 0)
    Sx = Sx / w
#    Sx = Sx[5:, 5:]
#    Sx = Sx[0, 0].reshape((1,1))
    Sx = Sx[:, 0].reshape((Sx.shape[0],1))
    Sx = np.array([[1]])
    H_lp = ms.TF([1], [1/wc**2, 1.4142/wc, 1])  # Low pass butterworth

    sA, sB, sC, sD = np.load('data/ss_param.npy')
    H_ring = ms.TF(sA, sB, sC, sD)
    H_ring.num = H_ring.num.real
    H_ring.num *= H_ring.den[-1]/H_ring.num[-1]
    H_ring.plot_hw(np.logspace(-1, 4)*2*pi, figsize=(4,3))
    sns.despine()
    plt.savefig('ctl_id_3z.pdf')
    H_ring.plot_step(figsize=(4,3))
    sns.despine()
    plt.grid('on')
    plt.savefig('ctl_id_3z_step.pdf')
    H_ring.num[0] = 0
    H_ring.plot_hw(np.logspace(-1, 4)*2*pi, figsize=(4,3))
    sns.despine()
    plt.savefig('ctl_id_2z.pdf')
    H_ring.plot_step(figsize=(4,3))
    plt.ylim(ymax=1.1)
    sns.despine()
    plt.grid('on')
    plt.savefig('ctl_id_2z_step.pdf')
    pid = ms.PID(0.6, 1.0*fs, 0.3/fs)
    pid = ms.PID(0, .8*fs, 0./fs)

    delay_calc = 2e-3  # 2e-3
    delay_adc = 0.5e-3
    delay_dac = 0.5e-3
    delay = delay_calc + delay_adc + delay_dac

    H_delay = ms.TF(*control.pade(delay))
    pid = ms.PID(0.9,0.5*fs, 0.15/fs)

#    H_delay = ms.TF([1],[1])
#    H = pid*H_delay*H_lp

    Hpid = pid*H_delay*H_lp
    Gpid = H_ring /(1 + Hpid*H_ring)
    Gpid.plot_hw(w=np.linspace(0.0001, 75)*2*np.pi, bode=False, xscale='linear', yscale='db', figsize=(4,3))
    sns.despine()
    plt.savefig('ctl_freqresp_pid.pdf')

    amplitude = 1
    perturbation = 'real'
    if perturbation == 'step':
        d = amplitude*ms.inputs.step(t, 0.1)
    elif perturbation == 'sinesweep':
        d = amplitude*ms.inputs.sinesweep(t, fmin=1e-3, fmax=75)
    elif perturbation == 'sine':
        fsin = 10
        d = amplitude*np.sin(2*np.pi*t*fsin)
    elif perturbation == "impulse":
        d = amplitude*ms.inputs.impulse(t, 0.1)
    elif perturbation == 'real':
        d = amplitude*ms.bessy.real_perturbation(t)

#    H_ring = ms.TF([1],[1])
#    H_dip.num *= H_dip.den[-1]/H_dip.num[-1]
    H_dip = ms.TF([1], [1])
#    delay = 0
#    H_ring.plot_step()
#    H_dip.plotStep()
#    H_ring.plot_hw(np.logspace(-1, 10))

#    H_dip.plotHw()

    freqs = np.fft.fftfreq(N, t[1])[:N//2]
    id50 = np.argmin(abs(freqs-50))
    plt.figure(figsize=(4,3))
#    plt.subplot(211)
#    plt.plot(freqs, np.abs(np.fft.fft(d))[:N//2]*2/N)
#    plt.title('Perturbation')
#    plt.xlabel('Frequency [in Hz]')
#    plt.ylabel('Amplitude (arbitrary unit)')
#    plt.grid('on')
#    plt.subplot(212)

    #for corrector in correctors[:-1:2]:
        #fb = corrector[0,0][0,0]
        #K = ms.TF(corrector[0,1][0,:], corrector[0,2][0,:])

        #zefunc(K,fb)
        #plt.draw()


    #plt.title(r'Output $x(t)$')
    #plt.xlabel('Frequency [in Hz]')
    #plt.ylabel('Amplitude (arbitrary unit)')
    #plt.legend(frameon=True, fancybox=True)
    #plt.tight_layout()
    #plt.grid('on')
    #sns.despine()

    #H = K*H_delay*H_lp
##    H = ms.PID(0, -0.8, 0)
    #G = H_ring /(1 + H*H_ring)
    #G.plot_hw(w=np.linspace(0.0001, 75)*2*np.pi, bode=False, xscale='linear', yscale='db', figsize=(4,3))
    #sns.despine()
    #plt.savefig('ctl_freqresp_hinf.pdf')


##    y, x, fs_r = ms.bessy.simulate(d, Kcor, Sx, H_lp, H_dip, H_ring, delay, fs, True)
    best_pid_by_hand = ms.PID(0.,0.8*fs, 0.)
#    best_pid_by_hand = ms.TF([1, 0.2*80],[1,0])*ms.TF([1/4000,1],[1/400,1])
    y, x, fs_r = ms.bessy.simulate(d, best_pid_by_hand, Sx, H_lp, H_dip, H_ring, delay, fs, True)
    plt.figure(figsize=(4,3))
    plt.subplot(2,1,1)
    value = np.abs(np.fft.fft(d))[:N//2][id50]
    plt.plot(np.fft.fftfreq(N, t[1])[:N//2], np.abs(np.fft.fft(d))[:N//2]/value, linewidth=1.3)
    #plt.title('Perturbation')
    plt.xlabel('Frequency [in Hz]')
    plt.ylabel('Amplitude\n(arbitrary unit)')
    plt.ylim([0,.7])
    plt.yticks(np.arange(0,.7,.2))
    sns.despine()
    plt.grid(which="both")

    value = np.abs(np.fft.fft(y[0,:]))[:N//2][id50]
    plt.subplot(2,1,2)
    plt.plot(np.fft.fftfreq(N, t[1])[:N//2], np.abs(np.fft.fft(y[0,:]))[:N//2]/value, linewidth=1.3)
    #plt.title(r'Output $x(t)$')
    plt.xlabel('Frequency [in Hz]')
    plt.ylabel('Amplitude\n(arbitrary unit)')
    plt.yticks(np.arange(0,2.1,.5))
    plt.ylim([0,2.1])
    sns.despine()
    plt.grid(which="both")
    plt.tight_layout()
    plt.savefig('ctl_sim_bestpid.pdf')

#    ms.TF_from_signal(y[0,:], x, fs_r, method='fft', plot=True, plottitle='with delay')
    plt.show()
