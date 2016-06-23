#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Bessy Signal module

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""
from __future__ import division, print_function

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.signal as signal


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
    xpid = np.zeros(CM_nb*(pid.den.size-1))

    sample = 0
    for k in range(1, t_real.size):
        d[k] = d_s[sample]

        # S* x delta_x
        dorbit, xlp = H_lp.apply_f(r-orbit[:, k-1], xlp, Ts_real)

        if t_real[k] >= t[sample] and sample < t.size-1:
            sample += 1
            e_s[:, sample] = S_inv.dot(dorbit).reshape(CM_nb)

#            du_s[:, sample] = pid.apply_fd(e_s[:, :sample+1], Ts)
            du_s[:, sample], xpid = pid.apply_f(e_s[:, sample], xpid, Ts)

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
        idx = np.argmax(np.linalg.norm(orbit, axis=1))
        plt.figure()
        plt.plot(t_real, d.T, label='perturbation')
        plt.plot(t_real, u[0, :].T, '-m', label='command (PID)')
        plt.plot(t_real, u_delay[0, :].T, '--c', label='delayed command (PID)')
        plt.plot(t_real, yd[0, :].T, '-r', label='output')
        plt.plot(t_real, orbit[idx, :].T, '-k', label='orbit')

#        plt.legend(loc='best')
        plt.title('Simulation result')

    return yd, d, fs_real


def toeplitz_block(col, row=None):
    if row is None:
        row = [col[0]]
        for elem in col:
            row.append(elem.conjugate())
    if len(col) != len(row):
        raise ValueError("Both args must have same length")
    if not np.all(col[0] == row[0]):
        raise ValueError("Both args must have same 1st element")
    shape = None
    for k in range(len(col)):
        if shape is not None:
            if col[k].shape != shape or row[k].shape != shape:
                raise ValueError("All elements must have same shape, given: "
                                 "{} and {}".format(row[k].shape, col[k].shape))
        shape = col[k].shape

    for elem in row:
        if elem.shape != shape:
            raise ValueError("All elements must have same shape")
        shape = elem.shape

    arraylist = [col[0]] * len(col)
    A = scipy.linalg.block_diag(*arraylist)

    for k in range(1, len(row)):
        print(k)
        arraylist = [col[k]] * (len(col)-k)
        A[k*shape[0]:, :-k*shape[1]] += scipy.linalg.block_diag(*arraylist)
#        A[k*shape[0]:, :-k*shape[1]] += np.kron(np.eye(len(col)-k), col[k])

#        arraylist = [row[k]] * (len(row)-k)
#        A[:-k*shape[0], k*shape[1]:] += scipy.linalg.block_diag(*arraylist)
#        A[:-k*shape[0], k*shape[1]:] += np.kron(np.eye(len(row)-k), row[k])

    return A


def control_toeplitz(H, Ts, N):
    if H.num.size == 1 and H.den.size == 1:
        return np.eye(N)*H.num[0]/H.den[0]

    A, B, C, D, _ = signal.cont2discrete((H.A, H.B, H.C, H.D), Ts)
    col = [D, C]
    for k in range(2, N):
        col.append(col[k-1].dot(A))
    for k in range(1, N):
        col[k] = col[k].dot(B)
    row = [D] + [np.zeros(D.shape)] * (N-1)
    print("toeplitz ready")
    return toeplitz_block(col, row)


def decimate(N_in, N_out):
    M = np.zeros(N_out, N_in)
    for k in range(N_out):
        M[k, (N_in//N_out)*k] = 1
    return M


def interpol(N_in, N_out):
    ratio = N_out//N_in
    M = np.zeros(N_out, N_in)

    for k in range(N_in):
        M[k*ratio:(k+1)*ratio, k] = np.ones(ratio)
    return M


def simulate_fast(d_s, pid, S, H_lp, H_dip, H_ring, delay=0, fs=1, plot=False):
    """
    The bloc diagram is the following:
    ::
                        . . . . . . . . . . .
                        . mBox              .
                        . Fs=150Hz          .
                        .                   .                              | d
     r=0  -     +----+  . +----+ e  +-----+ . u +-------+ ud  +------+  y  v    +-------+
      ---->(+)->| -\ |--->| S* |--->| PID |---->| delay |---->| Hdip |--->(+)-->| Hring |--+--> orbit
            ^   +----+  . +----+    +-----+ .   +-------+     +------+       yd +-------+  |
            |           . . . . . . . . . . .                                              |
            |                                                                              |
            +------------------------------------------------------------------------------+
               orbit

       ---Real-time----> <---Sampled-time---> <--Real-time -------
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

    print('Mring')
    Mring = control_toeplitz(H_ring, Ts_real, t_real.size)
    print('Mdip')
    Mdip = control_toeplitz(H_dip, Ts_real, t_real.size)
    print('delay')
    Mdelay = np.diag(np.ones(t_real.size-delay_offset), -delay_offset)
    print('pid')
    Mpid = control_toeplitz(pid, Ts, t.size)
    G = Mdip.dot(Mdelay.dot(interpol(t.size, t_real.size).dot(Mpid.dot(S_inv.dot(decimate(t_real.size, t.size))))))
    Mring.dot(np.inv((np.eye((t_real.size, t_real.size)) - G.dot(Mring))))

#
#    sample = 0
#    for k in range(1, t_real.size):
#        d[k] = d_s[sample]
#
#        # S* x delta_x
#        dorbit, xlp = H_lp.apply_f(r-orbit[:, k-1], xlp, Ts_real)
#
#        if t_real[k] >= t[sample] and sample < t.size-1:
#            sample += 1
#            e_s[:, sample] = S_inv.dot(dorbit).reshape(CM_nb)
#
##            du_s[:, sample] = pid.apply_f(e_s[:, :sample+1], Ts)
#            du_s[:, sample], xpid = pid.apply_fw(e_s[:, sample], xpid, Ts)
#
#            # Correction sent to PS
##            u_s[sample] = u_s[sample-1] + Ts*du_s[sample]
#        e[:, k] = e_s[:, sample]
#        u[:, k] = du_s[:, sample]
#
#        # Time for computation/PS
#        if k >= delay_offset:
#            u_delay[:, k] = u[:, k-delay_offset]
#
#        # Corrector magnet propagation
#        y[:, k], xcor = H_dip.apply_f(u_delay[:, k], xcor, Ts_real)
#        yd[:, k] = y[:, k] + d[k]
#
#        # Response of the ring
#        normalized_orbit, xring = H_ring.apply_f(yd[:, k], xring, Ts_real)
#        orbit[:, k] = S.dot(normalized_orbit).reshape(BPM_nb)
#
#    if plot:
#        idx = np.argmax(np.linalg.norm(orbit, axis=1))
#        plt.figure()
#        plt.plot(t_real, d.T, label='perturbation')
#        plt.plot(t_real, u[0, :].T, '-m', label='command (PID)')
#        plt.plot(t_real, u_delay[0, :].T, '--c', label='delayed command (PID)')
#        plt.plot(t_real, yd[0, :].T, '-r', label='output')
#        plt.plot(t_real, orbit[idx, :].T, '-k', label='orbit')
#
##        plt.legend(loc='best')
#        plt.title('Simulation result')
#
#    return yd, d, fs_real
