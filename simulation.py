#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
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


def model_1_delay(d_s, pid, H, H_ring, delay=0, fs=1):
    """
    The bloc diagram is the following:

                                                     | d
     r=0     e  +-----+ u  +-------+ ud  +-----+  y  v
    ---->(+)--->| PID |--->| delay |---->|  H  |--->(+)---+--> yd
        - ^     +-----+    +-------+     +-----+          |
          |                                               |
          +-----------------------------------------------+
             yd

     -----> <-Sample time->  <--real time -------
    """

    fs_real = 20*fs
    Ts_real = 1/fs_real
    t_real = np.linspace(0, t_max-Ts_real, fs_real*t_max)
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
    xcor = np.zeros(H.den.size-1)

    sample = 0
    for k in range(1, t_real.size):
        d[k] = d_s[sample]

        # S* x delta_x
        e[k] = coeff*(orbit[k-1]-r)
        if t_real[k] >= t[sample] and sample < t.size-1:
            sample += 1
            e_s[sample] = e[k]
            du_s[sample] = pid.apply_f(e_s[:sample+1])

            # Correction sent to PS
            u_s[sample] = u_s[sample-1] - du_s[sample]
        u[k] = u_s[sample]

        # Time for computation/PS
        if k >= delay_offset:
            u_delay[k] = u[k-delay_offset]

        # Corrector magnet propagation
        y[k], xcor = H.apply_f(u_delay[k], xcor, Ts_real)
        yd[k] = y[k] + d[k]

        # Response of the ring
        orbit[k], xring = H_ring.apply_f(yd[k], xring, Ts_real)

    plt.figure()
    plt.plot(t_real, d)
    plt.plot(t_real, u)
    plt.plot(t_real, u_delay, '-c')
    plt.plot(t_real, yd, '-r')
    plt.plot(t_real, e, '-m')
    plt.plot(t_real, orbit, '-k')

    plt.legend(['perturbation',
                'command (PID)',
                'delayed command (PID)',
                'output'
                ], loc='best')
    plt.title('With correction delay')

    return yd, d, fs_real

if __name__ == "__main__":

    plt.ion()
    plt.close('all')
    fs = 150.
    Ts = 1/fs
    t_max = 5.

    A = 3
    a = A / (2*np.pi*3)

    t = np.linspace(0, t_max-Ts, fs*t_max)

    """ H = A / (1 + a * s) """
    H = ms.TF([A], [a, 1])

#    print(H)
#    ring_data = np.load('ring_tf.npy')[0]
#    ndnum = np.flipud(np.array(ring_data['num'])*np.exp(1j*ring_data['phase']))
#    num = ndnum.tolist()
#    den = np.flipud(np.array(ring_data['den'])).tolist()
#    H_ring = ms.TF(num, den)
    sA, sB, sC, sD = np.load('ss_param.npy')
    H_ring = ms.TF(sA, sB, sC, sD)
    H_ring.num = -H_ring.num.real
#    H.plotHw(w=np.logspace(-1,3)*2*np.pi, ylabel="Peak current [in A]")
#    H_ring.plotHw()
    pid = ms.PID(0.8, 0, 0)

    G = 1/(1-H*pid)
#    G.plotHw(w=np.logspace(-1, 3)*2*np.pi)


#    d = np.zeros(t.size) ; d[np.argmin(abs(t-.1))] = 1 ; d[np.argmin(abs(t-(t[-1]-.2)))] = -1

    # Step
    d = np.piecewise(t, [t < 0.1, t >= 0.1, t > t_max-0.2], [0, 1, 0])
    amplitude = 0.02
    f1 = 1e-3
    f2 = 75
#    f_lin = 2*f1*t+(f2-f1)/t[-1]*0.5*(t**2)
#    f_exp = f1*t[-1]/np.log(f2/f1)*(np.exp(-t)-1)
#    d = amplitude*np.sin(2*np.pi*f_exp)

    # Sinus
#    d = amplitude*np.sin(2*np.pi*t*2.5)
#    y, x, fs_r = model_1(d, pid, H, fs)
#    ms.TF_from_signal(y, x, fs_r, plottitle='no delay')

    delay_calc = 1e-3  # 2e-3
    delay_adc = 1e-4
    delay_dac = 3e-4
    delay = delay_calc + delay_adc + delay_dac
#    y, x, fs_r = model_1_delay(d, pid, H, H_ring, 0, fs)

    pid = ms.PID(0.8,0,0)
    y, x, fs_r = model_1_delay(d, pid, H, H_ring, delay, fs)

#    ms.TF_from_signal(y, x, fs_r, plot=True, plottitle='with delay')

