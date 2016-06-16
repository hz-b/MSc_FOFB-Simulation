#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Bessy Signal module

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""
from __future__ import division, print_function

import math
import matplotlib.pyplot as plt
import numpy as np

import mysignal as ms
import search_kicks.tools as sktools

def corrector_order1():
    """ Create a transfer function of shape

    .. math::
        H = \\frac{A}{1 + a \cdot s}

    from measured frequency response.
    """
    A = 3
    a = A / (2*np.pi*3)
    return ms.TF([A], [a, 1])


def simulate(d_s, pid, S, H_lp, H_dip, H_ring, delay=0, fs=1, plot=False):
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

       ---Real-time----> <---------Sampled-time-----------> <--Real-time -------
    """

    f_ratio = 10
    fs_real = f_ratio*fs
    Ts_real = 1/fs_real
    Ts = 1/fs
    t_real = np.arange(0, f_ratio*d_s.size) / fs_real
    t = np.arange(0, d_s.size) / fs
    delay_offset = math.ceil(delay*fs_real)

    BPM_nb = S.shape[0]
    CM_nb = S.shape[1]

    svd_nb = min(S.shape[0], min(S.shape[1], 48))
    S_inv = sktools.maths.inverse_with_svd(S, svd_nb)

    # Init real time variables
    r = 0
    y = np.zeros((CM_nb, t_real.size))
    yd = np.zeros((CM_nb, t_real.size))
    orbit = np.zeros((BPM_nb, t_real.size))
    u = np.zeros((CM_nb, t_real.size))
    u_delay = np.zeros((CM_nb, t_real.size))
    d = np.zeros(t_real.size)
    e = np.zeros((CM_nb, t_real.size))

    # Init sample time variables
    u_s = np.zeros((CM_nb, t.size))
    du_s = np.zeros((CM_nb, t.size))
    e_s = np.zeros((CM_nb, t.size))

    xring = np.zeros(CM_nb*(H_ring.den.size-1))
    xcor = np.zeros(CM_nb*(H_dip.den.size-1))
    xlp = np.zeros(BPM_nb*(H_lp.den.size-1))
    xpid = np.zeros(pid.den.size-1)

    sample = 0
    for k in range(1, t_real.size):
        d[k] = d_s[sample]

        # S* x delta_x
        dorbit, xlp = H_lp.apply_f(r-orbit[:, k-1], xlp, Ts_real)

        if t_real[k] >= t[sample] and sample < t.size-1:
            sample += 1
            e_s[:, sample] = S_inv.dot(dorbit).reshape(CM_nb)

            du_s[:, sample] = pid.apply_f(e_s[:, :sample+1], Ts)
#            du_s[sample], xpid = pid.apply_fw(e_s[sample], xpid, Ts)

            # Correction sent to PS
#            u_s[sample] = u_s[sample-1] + Ts*du_s[sample]
        e[:, k] = e_s[:, sample]
        u[:, k] = du_s[:, sample]

        # Time for computation/PS
        if k >= delay_offset:
            u_delay[:, k] = u[:, k-delay_offset]

        # Corrector magnet propagation
        y[:, k], xcor = H_dip.apply_f(u_delay[:, k], xcor, Ts_real)
        yd[:, k] = y[:, k] + d[k]

        # Response of the ring
        normalized_orbit, xring = H_ring.apply_f(yd[:, k], xring, Ts_real)
        orbit[:, k] = S.dot(normalized_orbit).reshape(BPM_nb)

    if plot:
        plt.figure()
        plt.plot(t_real, d.T, label='perturbation')
        plt.plot(t_real, u.T, '-m', label='command (PID)')
        plt.plot(t_real, u_delay.T, '--c', label='delayed command (PID)')
        plt.plot(t_real, yd.T, '-r', label='output')
        plt.plot(t_real, orbit.T, '-k', label='orbit')

#        plt.legend(loc='best')
        plt.title('Simulation result')

    return yd, d, fs_real
