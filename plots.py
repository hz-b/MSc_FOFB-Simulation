#!/usr/bin/env python
# encode: utf8

from scipy.io import loadmat
from scipy.signal import freqresp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
plt.close('all')

f = loadmat('data/tf.mat')['f'][0, :]
rawdata = loadmat('data/tf.mat')['h0'][0, :]
A,B,C,D = np.load('data/ss_param.npy')
_, Hfit = freqresp((A,B,C,D), 2*np.pi*f)

plt.figure(figsize=(4,3))
plt.plot(f, abs(rawdata), '-b', label='Data')
plt.plot(f, abs(Hfit), '-r', label='Fitted data')
plt.ylabel('Amplitude [in mm/A]')
plt.xlabel('Frequency [in Hz]')
plt.yscale('log')
plt.xscale('log')
plt.grid(which='both')
plt.ylim([min(abs(rawdata)),11])
plt.xlim([0.3,f[-1]])
plt.legend(loc='best', frameon=True, fancybox=True)
sns.despine()
plt.tight_layout()
plt.savefig('ctl_id_amplitude.pdf')

plt.figure(figsize=(4,3))
plt.plot(f, np.unwrap(np.angle(rawdata))*180/np.pi, '-b', label='Data')
plt.plot(f, np.unwrap(np.angle(Hfit))*180/np.pi, '-r', label='Fitted data')
plt.ylabel('Phase [in deg]')
plt.xlabel('Frequency [in Hz]')
plt.xscale('log')
plt.grid(which='both')
plt.ylim(80,200)
plt.xlim([0.3,f[-1]])
plt.legend(loc='best', frameon=True, fancybox=True)
sns.despine()
plt.tight_layout()
plt.savefig('ctl_id_phase.pdf')

plt.show()
