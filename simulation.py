#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from math import pi
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import mysignal as ms
sys.path.append('../search_kicks')
import search_kicks.tools as sktools


def model_1(d, PID, H, fs):
    """
    The bloc diagram is the following:
    ::
                                        | d
      r=0     e  +-----+ u  +-----+ y   v  y
     ---->(+)--->| PID ]--->|  H  |--->(+)---+--> yd
         - ^     +-----+    +-----+          |
           |                                 |
           +---------------------------------+
              yd


    """

    r = 0
    N = d.size
    y = np.zeros(N)

    yd = np.zeros(N)
    e = np.zeros(N)
    u = np.zeros(N)
    xcor = np.zeros(H.den.size-1)
    for k in range(1, N):
        e[k] = r - yd[k-1]
        u[k] = pid.apply_f(e[:k+1])
#        u[k] = super(PID, pid).apply_f(e[:k+1],u[:k],Ts)
        y[k], xcor = H.apply_f(u[k], xcor, 1/fs)
        yd[k] = y[k] + d[k]

    plt.figure()
    t = np.arange(N)/fs
    plt.plot(t, d)
    plt.plot(t, u)
    plt.plot(t, yd)
    plt.legend(['perturbation', 'PID', 'output'], loc='best')
    plt.title('Without correction delay')

    return yd, d, fs


def model_1_delay(d_s, pid, H_dip, H_ring, delay=0, fs=1):
    """
    The bloc diagram is the following:
    ::
                        . . . . . . . . . . . . . . . . . .
                        . mBox                    +---+   .
                        . Fs=150Hz             +-| T |-+ .
                        .                       | +---+ | .                              | d
     r=0  -     +----+  . +----+ e  +-----+ du  v       | . u +-------+ ud  +------+  y  v    +-------+
      ---->(+)->| -\ |--->| S* |--->| PID |--->(+)------+---->| delay |---->| Hdip |--->(+)-->| Hring |--+--> orbit
            ^   +----+  . +----+    +-----+  -            .   +-------+     +------+       yd +-------+  |
            |           . . . . . . . . . . . . . . . . . .                                              |
            |                                                                                            |
            +--------------------------------------------------------------------------------------------+
               orbit

       ---real-time----> <---------Sampled-time-----------> <--real-time -------
    """

    f_ratio = 10
    fs_real = f_ratio*fs
    Ts_real = 1/fs_real
    Ts = 1/fs
    t_real = np.arange(0, f_ratio*d_s.size) / fs_real
    t = np.arange(0, d_s.size) / fs
    delay_offset = math.ceil(delay*fs_real)

    coeff = H_ring.den[-1]/H_ring.num[-1]

    # Init real time variables
    r = 0
    y = np.zeros(t_real.size)
    yd = np.zeros(t_real.size)
    orbit = np.zeros(t_real.size)
    u = np.zeros(t_real.size)
    u_delay = np.zeros(t_real.size)
    d = np.zeros(t_real.size)
    e = np.zeros(t_real.size)

    # Init sample time variables
    u_s = np.zeros(t.size)
    du_s = np.zeros(t.size)
    e_s = np.zeros(t.size)

    xring = np.zeros(H_ring.den.size-1)
    xcor = np.zeros(H_dip.den.size-1)
    xlp = np.zeros(H_lp.den.size-1)
    xpid = np.zeros(pid.den.size-1)

    sample = 0
    for k in range(1, t_real.size):
        d[k] = d_s[sample]

        # S* x delta_x
        dorbit, xlp = H_lp.apply_f(r-orbit[k-1], xlp, Ts_real)
        e[k] = coeff*(dorbit)

        if t_real[k] >= t[sample] and sample < t.size-1:
            sample += 1
            e_s[sample] = e[k]
            du_s[sample] = pid.apply_f(e_s[:sample+1], Ts)
#            du_s[sample], xpid = pid.apply_fw(e_s[sample], xpid, Ts)

            # Correction sent to PS
#            u_s[sample] = u_s[sample-1] + Ts*du_s[sample]
        u[k] = du_s[sample]

        # Time for computation/PS
        if k >= delay_offset:
            u_delay[k] = u[k-delay_offset]

        # Corrector magnet propagation
        y[k], xcor = H_dip.apply_f(u_delay[k], xcor, Ts_real)
        yd[k] = y[k] + d[k]

        # Response of the ring
        orbit[k], xring = H_ring.apply_f(yd[k], xring, Ts_real)

    plt.figure()
    plt.plot(t_real, d, label='perturbation')
#    plt.plot(np.linspace(0, t_max-Ts, fs*t_max), du_s, label='command (PID)')
    plt.plot(t_real, u, '-m', label='command (PID)')
    plt.plot(t_real, u_delay, '--c', label='delayed command (PID)')
    plt.plot(t_real, yd, '-r', label='output')
    plt.plot(t_real, orbit, '-k', label='orbit')

    plt.legend(loc='best')
    plt.title('With correction delay')

    return yd, d, fs_real


if __name__ == "__main__":

    plt.ion()
    plt.close('all')
    fs = 150.
    Ts = 1/fs
    t_max = 60

    A = 3
    a = A / (2*np.pi*3)

    t = np.linspace(0, t_max-Ts, fs*t_max)

    """ H = A / (1 + a * s) """
    H_dip = ms.TF([A], [a, 1])

    wc = 2*pi*80
    H_lp = ms.TF([1], [1/wc**2, 1.4142/wc, 1])  # Low pass butterworth

    sA, sB, sC, sD = np.load('ss_param.npy')
    H_ring = ms.TF(sA, sB, sC, sD)
    H_ring.num = -H_ring.num.real
    pid = ms.PID(0, 0.8*fs, 0)

    delay_calc = 1e-3  # 2e-3
    delay_adc = 1e-4
    delay_dac = 3e-4
    delay = delay_calc + delay_adc + delay_dac

#    S_coeff = H_ring.den[-1]/H_ring.num[-1]
#    H = S_coeff*H_lp*H_dip*pid
#    H = ms.PID(0, -0.8, 0)
#    G = H_ring /(1 - H*H_ring)
#    G.plotHw(w=np.logspace(-1, 2)*2*np.pi, bode=True)

#    d = np.zeros(t.size) ; d[np.argmin(abs(t-.1))] = 1 ; d[np.argmin(abs(t-(t[-1]-.2)))] = -1

    # Step
    amplitude = 0.02
#    d = amplitude*np.piecewise(t, [t < 0.1, t >= 0.1, t > t_max-0.2], [0, 1, 0])
#    d = amplitude*np.piecewise(t, [t < 0.1, t >= 0.1], [0, 1])
    f1 = 1e-3
    f2 = 75
    f_func = 2*f1*t+(f2-f1)/t[-1]*0.5*(t**2)
#    f_func = f1*Ts/np.log(f2/f1)*(np.exp(-t)-1)
    d = amplitude*np.sin(2*np.pi*f_func)

    # Sinus
#    d = amplitude*np.sin(2*np.pi*t*2.5)
#    y, x, fs_r = model_1(d, pid, H, fs)
#    ms.TF_from_signal(y, x, fs_r, plottitle='no delay')

#    H_ring = ms.TF([1],[1])
#    H_dip.num *= H_dip.den[-1]/H_dip.num[-1]
    H_dip =  ms.TF([1], [1])
#    delay = 0
#    H_ring.plotStep()
#    H_dip.plotStep()
#    H_dip.plotHw()


    y, x, fs_r = model_1_delay(d, pid, H_dip, H_ring, delay, fs)

    ms.TF_from_signal(y, x, fs_r, method='fft', plot=True, plottitle='with delay')

