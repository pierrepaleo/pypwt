#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
try:
    import pywt
except ImportError:
    print("ERROR : could not find the python module pywt. Please install it (sudo apt-get install python-pywt or http://www.pybytes.com/pywavelets/")
    exit(1)
try:
    import pypwt
except ImportError:
    print("ERROR: could not load pypwt. Make sure it is installed (python setup.py install --user)")
    exit(1)
# FIXME : use other images
try:
    from scipy.misc import lena
except ImportError:
    print("ERROR: could not load lena from scipy.misc")
    exit(1)


# See ppdwt/filters.h
available_filters = [
    "haar",
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
    "coif5",
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
    "rbior1.3",
    "rbior1.5",
    "rbior2.2",
    "rbior2.4",
    "rbior2.6",
    "rbior2.8",
    "rbior3.1",
    "rbior3.3",
    "rbior3.5",
    "rbior3.7",
    "rbior3.9",
    "rbior4.4",
    "rbior5.5",
    "rbior6.8"]




def print_errors(arr1, arr2, string=None):
    # TODO: shape check
    if string is None: string = ""
    print("%s max error: %e" % (string, np.max(np.abs(arr1 - arr2))))


def compare_coeffs(W, Wpy, swt=False):
    """
    Compares the coefficients of pydwt to those of pywt.

    W: pypwt.Wavelets instance
    Wpy: pywt coefficients (wavedec2 or swt2)
    swt: boolean
    """

    print("%s Running test for %s, %d levels %s" % ("--"*10, str(W.wname), W.levels, "-"*10))
    # retrieve all coefficients from GPU
    W_coeffs = W.coeffs

    if not(swt): # standard DWT
        levels = len(Wpy)-1
        if (levels != W.levels): raise ValueError("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
        A = Wpy[0]
        print_errors(A, W_coeffs[0], "[app]")
        for i in range(levels):
            D1, D2, D3 = Wpy[levels-i][0], Wpy[levels-i][1], Wpy[levels-i][2]
            """
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(W_coeffs[i+1][0])
            plt.colorbar()
            plt.show()
            """
            print_errors(D1, W_coeffs[i+1][0], "[det.H]")




    else: # SWT
        levels = len(Wpy)
        if (levels != W.levels): raise ValueError("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
        for i in range(levels):
            A, D1, D2, D3 = Wpy[levels-1-i][0], Wpy[levels-1-i][1][0], Wpy[levels-1-i][1][1], Wpy[levels-1-i][1][2]










if __name__ == '__main__':

    from pypwt import Wavelets
    wname = "db3"
    levels = 2
    do_swt = 0


    l = lena()
    W = Wavelets(l, wname, levels, do_swt=do_swt)
    W.forward()
    if not(do_swt):
        Wpy = pywt.wavedec2(l, wname, mode="per", level=levels)
    else:
        Wpy = pywt.swt2(l, wname, levels)

    compare_coeffs(W, Wpy, swt=bool(do_swt))







