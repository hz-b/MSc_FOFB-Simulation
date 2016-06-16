from __future__ import division, print_function

import numpy as np


def step(t, t0, amplitude=1):
    return amplitude*np.piecewise(t, [t < t0, t >= t0], [0, 1])


def sinesweep(t, fmin, fmax, amplitude=1, which='linear'):
    if which == 'linear':
        ft = 2*fmin*t+(fmax-fmin)/t[-1]*0.5*(t**2)
    elif which == 'exp':
        ft = fmin*t[-1]/np.log(fmax/fmin)*(np.exp(-t)-1)
    else:
        raise ValueError("Last argummet (which) should be 'exp' or 'linear', "
                         "'{}' given".format(which))

    return amplitude*np.sin(2*np.pi*ft)


def impulse(t, t0, amplitude=1):
    d = np.zeros(t.size)
    d[np.argmin(abs(t-t0))] = 1
    return d
