#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Short description.

@author: Olivier CHURLAUD <olivier.churlaud@helmholtz-berlin.de>
"""
from __future__ import division, print_function

import struct
import numpy as np

ADC_POS = 0x01000000
DAC_POS = 0x02000000
CTRL_POS = 0x03000000
CONF_POS = CTRL_POS + 1000
FILENAME = "../fofb/mBox++/build/dump_rmf.dat"
cf = 0.3051758e-3
halfDigits = 1 << 23


class ADC:
    LUT = dict()
    gains = dict()
    bpm_offsets = dict()

    def __init__(self):
        self.LUT['x'] = (read_from_struct('ADC_BPMIndex_PosX')[:, 0] - 1) \
            .astype(int).tolist()
        self.LUT['y'] = (read_from_struct('ADC_BPMIndex_PosY')[:, 0] - 1) \
            .astype(int).tolist()
        self.gains['x'] = read_from_struct('GainX')[:, 0]
        self.gains['y'] = read_from_struct('GainY')[:, 0]

        self.bpm_offsets['x'] = read_from_struct('BPMoffsetX')[:, 0]
        self.bpm_offsets['y'] = read_from_struct('BPMoffsetY')[:, 0]

    def index(self, xy, i):
        if xy not in ['x', 'y']:
            ValueError("1st arg must be x or y, not {}".format(xy))
        return self.LUT[xy][i]

    def read_BPMs(self):
        adc_values = read_adc()
        BPMx = adc_values[self.LUT['x']]
        BPMy = adc_values[self.LUT['y']]
        BPMx = BPMx * self.gains['x'] * cf * (-1) - self.bpm_offsets['x']
        BPMy = BPMy * self.gains['y'] * cf * (+1) - self.bpm_offsets['y']

        HBP2D6R = adc_values[160] * cf * 0.8
        BPMx[self.LUT['y'].index(163)] -= (-0.325 * HBP2D6R)

        HBP1D5R = adc_values[142] * cf * 0.8
        BPMx[self.LUT['x'].index(123-1)] -= (-0.42 * HBP1D5R)
        BPMx[self.LUT['x'].index(125-1)] -= (-0.84 * HBP1D5R)
        BPMx[self.LUT['x'].index(129-1)] -= (+0.84 * HBP1D5R)
        BPMx[self.LUT['x'].index(131-1)] -= (+0.42 * HBP1D5R)
        return BPMx, BPMy

    def write_BPMs(self, BPMx, BPMy):
        if BPMx.size != len(self.LUT['x']):
            ValueError("2nd arg must be have size of {}, not {}"
                       .format(len(self.LUT['x']), BPMx.size))
        if BPMy.size != len(self.LUT['y']):
            ValueError("3rd arg must be have size of {}, not {}"
                       .format(len(self.LUT['y']), BPMy.size))

        adc_values = read_adc()

        # Do  the opposite of read
        BPMy_adc = (BPMy + self.bpm_offsets['y'])*(+1) / cf / self.gains['y']
        for i in range(BPMy.size):
            adc_values[self.LUT['y'][i]] = BPMy_adc[i]

        HBP2D6R = adc_values[160] * cf * 0.8
        BPMx[self.LUT['x'].index(163-1)] += (-0.325 * HBP2D6R)

        HBP1D5R = adc_values[142] * cf * 0.8
        BPMx[self.LUT['x'].index(123-1)] += (-0.42 * HBP1D5R)
        BPMx[self.LUT['x'].index(125-1)] += (-0.84 * HBP1D5R)
        BPMx[self.LUT['x'].index(129-1)] += (+0.84 * HBP1D5R)
        BPMx[self.LUT['x'].index(131-1)] += (+0.42 * HBP1D5R)

        BPMx_adc = (BPMx + self.bpm_offsets['x'])*(-1) / cf / self.gains['x']

        for i in range(BPMx.size):
            adc_values[self.LUT['x'][i]] = BPMx_adc[i]

        write_adc(adc_values)


class DAC:
    LUT = dict()
    scaleDigits = dict()

    def __init__(self):
        self.LUT['x'] = (read_from_struct('DAC_HCMIndex')[:, 0] - 1) \
            .astype(int).tolist()
        self.LUT['y'] = (read_from_struct('DAC_VCMIndex')[:, 0] - 1)\
            .astype(int).tolist()

        self.scaleDigits['x'] = read_from_struct('scaleDigitsH')[:, 0]
        self.scaleDigits['y'] = read_from_struct('scaleDigitsV')[:, 0]

    def index(self, key, i):
        if key not in ['x', 'y']:
            ValueError("1st arg must be x or y, not {}".format(key))
        return self.LUT[key][i]

    def read_CMs(self):
        dac_values = read_dac()
        CMx = (dac_values[self.LUT['x']] - halfDigits) / self.scaleDigits['x']
        CMy = (dac_values[self.LUT['y']] - halfDigits) / self.scaleDigits['y']
        return CMx, CMy

    def write_CMs(self, CMx, CMy):
        if CMx.size != len(self.LUT['x']):
            ValueError("2nd arg must be have size of {}, not {}"
                       .format(len(self.LUT['x']), CMx.size))
        if CMy.size != len(self.LUT['y']):
            ValueError("3rd arg must be have size of {}, not {}"
                       .format(len(self.LUT['y']), CMy.size))

        CMx_dac = (CMx * self.scaleDigits['x']) + halfDigits
        CMy_dac = (CMy * self.scaleDigits['y']) + halfDigits

        dac_values = read_dac()

        for i in range(CMx.size):
            dac_values[self.LUT['x'][i]] = CMx_dac[i]

        for i in range(CMy.size):
            dac_values[self.LUT['y'][i]] = CMy_dac[i]

        write_dac(dac_values)


def read_adc():
    with open(FILENAME, 'rb') as f:
        f.seek(ADC_POS)
        return np.fromstring(f.read(255*8))  # 253 elements in double


def write_adc(values):
    with open(FILENAME, 'rb+') as f:
        f.seek(ADC_POS)
        f.write(values.tobytes())  # 115 elements in double


def read_dac():
    with open(FILENAME, 'rb') as f:
        f.seek(DAC_POS)
        return np.fromstring(f.read(115*8))  # 115 elements in double


def write_dac(values):
    with open(FILENAME, 'rb+') as f:
        f.seek(DAC_POS)
        f.write(values.tobytes())  # 115 elements in double


def read_from_struct(name):
    with open(FILENAME, 'rb') as f:
        f.seek(CONF_POS)
        element_nb = struct.unpack('h', f.read(2))[0]
        for element in range(element_nb):
            elem_name, row_nb, col_nb, elem_type = _read_from_struct_header(f)

            if elem_name == name:
                return _read_from_struct_value(f, row_nb, col_nb, elem_type)

            else:
                if elem_type == 1:
                    unit_size = 8
                else:
                    unit_size = 1
                f.read(row_nb*col_nb*unit_size)  # go to next header

        print("{} was not found in the struct".format(name))


def dump_struct():
    d = {}
    with open(FILENAME, 'rb') as f:
        f.seek(CONF_POS)
        element_nb = struct.unpack('h', f.read(2))[0]
        for element in range(element_nb):
            name, row_nb, col_nb, elem_type = _read_from_struct_header(f)
            d[name] = _read_from_struct_value(f, row_nb, col_nb, elem_type)
    return d


def write_struct(d):
    with open(FILENAME, 'rb+') as f:
        f.seek(CONF_POS)
        element_nb = len(d)
        f.write(struct.pack('h', element_nb))
        for key in d.keys():
            if type(d[key]) == float:
                row_nb = 1
                col_nb = 1
                elem_type = 1
            elif type(d[key]) == np.ndarray:
                shape = d[key].shape
                if len(shape) == 1:
                    row_nb = shape[0]
                    col_nb = 1
                    binary_value = struct.pack('d', d[key])
                else:
                    row_nb = shape[0]
                    col_nb = shape[1]
                elem_type = 1
                binary_value = d[key].tobytes()

            else:
                row_nb = 1
                col_nb = 1
                elem_type = 2
                binary_value = struct.pack('c', d[key])
            f.write(struct.pack('h', len(key)))   # name size
            f.write(struct.pack('h', row_nb))     # nb of rows
            f.write(struct.pack('h', col_nb))     # nb of cols
            f.write(struct.pack('h', elem_type))  # type of elem
            f.write(struct.pack(str(len(key))+'c',
                                key.encode('utf-8')))  # name
            f.write(binary_value)


def _read_from_struct_header(f):
    name_size = struct.unpack('h', f.read(2))[0]
    row_nb = struct.unpack('h', f.read(2))[0]
    col_nb = struct.unpack('h', f.read(2))[0]
    elem_type = struct.unpack('h', f.read(2))[0]

    name = struct.unpack(str(name_size)+'s',
                         f.read(name_size))[0]
    return name.decode('utf-8'), row_nb, col_nb, elem_type


def _read_from_struct_value(f, row_nb, col_nb, elem_type):
    if elem_type == 1:
        unit_size = 8  # double
    else:
        unit_size = 1  # char

    binary_size = row_nb*col_nb*unit_size
    binary_value = f.read(binary_size)
    if row_nb == 1 and col_nb == 1:
        if elem_type == 1:
            return struct.unpack('d', binary_value)[0]
        elif elem_type == 2:
            return struct.unpack('c', binary_value)[0]

    else:
        if elem_type == 1:
            value = np.fromstring(binary_value)
            return value.reshape((row_nb, col_nb))
        else:
            value = struct.unpack(str(binary_size)+'s',
                                  binary_value)[0]
            return value.decode('utf-8')
