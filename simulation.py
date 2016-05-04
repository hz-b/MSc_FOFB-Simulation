#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mysignal import *


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
    f_cont = fs*10
    t_cont = np.linspace(0, t_max-1/f_cont, f_cont*t_max)

    r = 0
    y = np.zeros(t.size)

    yd = np.zeros(t.size)
    e = np.zeros(t.size)
    u = np.zeros(t.size)

    for k in range(1, t.size):
        e[k] = r  - yd[k-1]
        u[k] = pid.apply_f(e[:k+1])
#        u[k] = super(PID, pid).apply_f(e[:k+1],u[:k],Ts)
        y[k] = H.apply_f(u[:k+1], y[:k], Ts)
        yd[k] = y[k] + d[k]

    plt.figure()
    plt.plot(t, d)
    plt.plot(t, u)
    plt.plot(t, yd)
    plt.legend(['perturbation','PID', 'output'], loc='best')

def model_1_delay(d):
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
        e[k] = r  - yd[k-1]
        u[k] = pid.apply_f(e[:k+1])
#        u[k] = super(PID, pid).apply_f(e[:k+1],u[:k],Ts)
        y[k] = H.apply_f(u[:k+1], y[:k], Ts)
        yd[k] = y[k] + d[k]

    plt.figure()
    plt.plot(t, d)
    plt.plot(t, u)
    plt.plot(t, yd)
    plt.legend(['perturbation','PID', 'output'], loc='best')

if __name__=="__main__":

    plt.ion()
    plt.close('all')
    fs = 150.
    Ts = 1/fs
    t_max = 1.

    A = 3
    a = A / (2*np.pi*3)

    t = np.linspace(0, t_max-Ts, fs*t_max)

    H = TF([A],[a,1])
    #H.plotHw(w=np.logspace(-1,3)*2*np.pi, ylabel="Peak current [in A]")

    pid = PID(1, 0.5, 0.5)

    G = 1/(1-H*pid)
    G.plotHw(w=np.logspace(-1,3)*2*np.pi)

#    d = np.zeros(t.size) ; d[np.argmin(abs(t-.1))] = 1 ; d[np.argmin(abs(t-(t[-1]-.2)))] = -1

    # Step
    d = np.piecewise(t, [t < 0.1, t >= 0.1, t > t_max-0.2], [0, 1, 0])
    model_1(d)

    # Sinus
    d = np.sin(2*np.pi*t*2.5)
    model_1(d)
