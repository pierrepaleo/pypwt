#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pywt import wavedec, wavedec2, waverec, waverec2, swt, swt2
try:
    from pywt import iswt, iswt2
except ImportError: # nigma/pywt
    iswt = None
    iswt2 = None

try:
    from scipy.misc import ascent
    scipy_img = ascent()
except ImportError:
    from scipy.misc import lena
    scipy_img = lena()


def iDivUp(a, b):
    return (a + (b - 1))//b



def create_data_to_good_size(data, size):
    """
    From a numpy array, create a second numpy array with a given size.
    The result contains the tiled data, which is then cropped to the wanted size.
    """
    # For 1D
    if min(size) == 1:
        sz = max(size)
        clip = iDivUp(sz, data.shape[0])
        res = np.tile(data[0], clip)[:sz]
        return res[:, np.newaxis].T

    clip_r = iDivUp(size[0], data.shape[0])
    clip_c = iDivUp(size[1], data.shape[1])
    res = np.tile(data, (clip_r, clip_c))
    return res[:size[0], :size[1]]



what_to_params = {
    "dwt2": {
        "name": "2D Forward DWT",
        "do_swt": 0,
        "ndim": 2,
        "pywt_function": wavedec2,
     },
    "idwt2": {
        "name": "2D Inverse DWT",
        "do_swt": 0,
        "ndim": 2,
        "pywt_function": waverec2,
     },
    "dwt": {
        "name": "1D Forward DWT",
        "do_swt": 0,
        "ndim": 1,
        "pywt_function": wavedec,
     },
    "idwt": {
        "name": "1D Inverse DWT",
        "do_swt": 0,
        "ndim": 1,
        "pywt_function": waverec,
     },
    "batched dwt": {
        "name": "Batched 1D Forward DWT",
        "do_swt": 0,
        "ndim": 1,
        "pywt_function": wavedec,
     },
    "batched idwt": {
        "name": "Batched 1D Inverse DWT",
        "do_swt": 0,
        "ndim": 1,
        "pywt_function": waverec,
     },
    "swt2": {
        "name": "2D Forward SWT",
        "do_swt": 1,
        "ndim": 2,
        "pywt_function": swt2,
     },
    "iswt2": {
        "name": "2D Inverse SWT",
        "do_swt": 1,
        "ndim": 2,
        "pywt_function": iswt2,
     },
    "swt": {
        "name": "1D Forward SWT",
        "do_swt": 0,
        "ndim": 2,
        "pywt_function": swt,
     },
    "iswt": {
        "name": "1D Inverse SWT",
        "do_swt": 1,
        "ndim": 1,
        "pywt_function": iswt,
     },
    "batched swt": {
        "name": "Batched 1D Forward SWT",
        "do_swt": 1,
        "ndim": 1,
        "pywt_function": swt,
     },
    "batched iswt": {
        "name": "Batched 1D Inverse SWT",
        "do_swt": 1,
        "ndim": 1,
        "pywt_function": iswt,
     },
}




# See ppdwt/filters.h
available_filters = [
    "haar",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "db9",
    "db10",
    "db11",
    "db12",
    "db13",
    "db14",
    "db15",
    "db16",
    "db17",
    "db18",
    "db19",
    "db20",
    "sym2",
    "sym3",
    "sym4",
    "sym5",
    "sym6",
    "sym7",
    "sym8",
    "sym9",
    "sym10",
    "sym11",
    "sym12",
    "sym13",
    "sym14",
    "sym15",
    "sym16",
    "sym17",
    "sym18",
    "sym19",
    "sym20",
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5", # pywt 0.5 has a problem with this one
    "bior1.3",
    "bior1.5",
    "bior2.2",
    "bior2.4",
    "bior2.6",
    "bior2.8",
    "bior3.1",
    "bior3.3",
    "bior3.5",
    "bior3.7",
    "bior3.9",
    "bior4.4",
    "bior5.5",
    "bior6.8",
    "rbio1.3",
    "rbio1.5",
    "rbio2.2",
    "rbio2.4",
    "rbio2.6",
    "rbio2.8",
    "rbio3.1",
    "rbio3.3",
    "rbio3.5",
    "rbio3.7",
    "rbio3.9",
    "rbio4.4",
    "rbio5.5",
    "rbio6.8"]
# ------
