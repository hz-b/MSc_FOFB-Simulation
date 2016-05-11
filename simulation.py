#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import mysignal as ms


def model_1(d):
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
    y = np.zeros(t.size)

    yd = np.zeros(t.size)
    e = np.zeros(t.size)
    u = np.zeros(t.size)

    for k in range(1, t.size):
        e[k] = r - yd[k-1]
        u[k] = pid.apply_f(e[:k+1])
#        u[k] = super(PID, pid).apply_f(e[:k+1],u[:k],Ts)
        y[k] = H.apply_f(u[:k+1], y[:k], Ts)
        yd[k] = y[k] + d[k]

    plt.figure()
    plt.plot(t, d)
    plt.plot(t, u)
    plt.plot(t, yd)
    plt.legend(['perturbation', 'PID', 'output'], loc='best')
    plt.title('Without correction delay')


def model_1_delay(d_s, delay=0):
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

    # Init real time variables
    r = 0
    y = np.zeros(t_real.size)
    yd = np.zeros(t_real.size)
    u = np.zeros(t_real.size)
    u_delay = np.zeros(t_real.size)
    d = np.zeros(t_real.size)
    e = np.zeros(t_real.size)

    # Init sample time variables
    u_s = np.zeros(t.size)
    e_s = np.zeros(t.size)

    sample = 0
    for k in range(1, t_real.size):

        e[k] = r - yd[k-1]
        d[k] = d_s[sample]
        if t_real[k] >= t[sample] and sample < t.size-1:
            sample += 1
            e_s[sample] = e[k]
            u_s[sample] = pid.apply_f(e_s[:sample+1])
#            u_s[last] = super(ms.PID, pid).apply_f(e_s[:last+1],u[:last],Ts)
        u[k] = u_s[sample]
        if k >= delay_offset:
            u_delay[k] = u[k-delay_offset]

        y[k] = H.apply_f(u_delay[:k+1], y[:k], Ts_real)
        yd[k] = y[k] + d[k]

    plt.figure()
    plt.plot(t_real, d)
    plt.plot(t_real, u)
    plt.plot(t_real, u_delay, '-c')
    plt.plot(t_real, yd, '-r')
    plt.legend(['perturbation',
                'command (PID)',
                'delayed command (PID)',
                'output'
                ], loc='best')
    plt.title('With correction delay')

if __name__ == "__main__":

    plt.ion()
    plt.close('all')
    fs = 150.
    Ts = 1/fs
    t_max = 1.

    A = 3
    a = A / (2*np.pi*3)

    t = np.linspace(0, t_max-Ts, fs*t_max)

    """ H = A / (1 + a * s) """
    H = ms.TF([A], [a, 1])
#   H.plotHw(w=np.logspace(-1,3)*2*np.pi, ylabel="Peak current [in A]")

    pid = ms.PID(0.8, 0.5, 0.5)

    G = 1/(1-H*pid)
    G.plotHw(w=np.logspace(-1, 3)*2*np.pi)

#    d = np.zeros(t.size) ; d[np.argmin(abs(t-.1))] = 1 ; d[np.argmin(abs(t-(t[-1]-.2)))] = -1

    # Step
    d = np.piecewise(t, [t < 0.1, t >= 0.1, t > t_max-0.2], [0, 1, 0])
    model_1(d)

    delay_calc = 1e-3  # 2e-3
    delay_adc = 1e-4
    delay_dac = 3e-4
    delay = delay_calc + delay_adc + delay_dac
    model_1_delay(d, delay)

    # Sinus
#    d = np.sin(2*np.pi*t*2.5)
#    model_1(d)
