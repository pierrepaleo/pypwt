#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
import logging
from time import time
try:
    import pywt
except ImportError:
    print("ERROR : could not find the python module pywt. Please install it (sudo apt-get install python-pywt or http://www.pybytes.com/pywavelets/")
    exit(1)
try:
    from pypwt import Wavelets
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


def elapsed_ms(t0):
    return (time()-t0)*1e3

def _calc_errors(arr1, arr2, string=None):
    if string is None: string = ""
    msg = str("%s max error: %e" % (string, np.max(np.abs(arr1 - arr2))))
    logging.info(msg)





# http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parametrized should
        inherit from this class.
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parametrize(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite



class TestWavelet(ParametrizedTestCase):#(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(filename='results.log', filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S', level=logging.DEBUG)
        self.tol = 1e-3 # Maximum acceptable error wrt pywt for float32 precision
        self.data = lena()
        # Default arguments when testing only one wavelet
        self.wname = "haar"
        self.levels = 8
        self.do_swt = 0


    def compare_coeffs(self, W, Wpy, swt=False):
        """
        Compares the coefficients of pydwt to those of pywt.

        W: pypwt.Wavelets instance
        Wpy: pywt coefficients (wavedec2 or swt2)
        swt: boolean
        """
        # retrieve all coefficients from GPU
        W_coeffs = W.coeffs

        if not(swt): # standard DWT
            levels = len(Wpy)-1
            if (levels != W.levels):
                err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
                logging.error(err_msg)
                raise ValueError(err_msg)
            A = Wpy[0]
            _calc_errors(A, W_coeffs[0], "[app]")
            for i in range(levels):
                D1, D2, D3 = Wpy[levels-i][0], Wpy[levels-i][1], Wpy[levels-i][2]
                logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
                if (W.wname == "haar"): # FIXME : in haar implementation, the detail coefficients are shifted
                    tmp = D2
                    D2 = D1
                    D1 = tmp
                # ----
                _calc_errors(D1, W_coeffs[i+1][0], "[det.H]")
                _calc_errors(D2, W_coeffs[i+1][1], "[det.V]")
                _calc_errors(D3, W_coeffs[i+1][2], "[det.D]")

        else: # SWT
            levels = len(Wpy)
            if (levels != W.levels):
                err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
                logging.error(err_msg)
                raise ValueError(err_msg)
            for i in range(levels):
                A, D1, D2, D3 = Wpy[levels-1-i][0], Wpy[levels-1-i][1][0], Wpy[levels-1-i][1][1], Wpy[levels-1-i][1][2]
                logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
                _calc_errors(D1, W_coeffs[i+1][0], "[det.H]")
                _calc_errors(D2, W_coeffs[i+1][1], "[det.V]")
                _calc_errors(D3, W_coeffs[i+1][2], "[det.D]")


    def test_wavelet(self):
        if self.param is None:
            wname = self.wname
            levels = self.levels
            do_swt = self.do_swt
        else:
            wname = self.param[0]
            levels = self.param[1]
            do_swt = self.param[2]

        logging.info("%s Running test for %s, %d levels %s" % ("-"*10, wname, levels, "-"*10))
        logging.info("SWT is %s" % ((["OFF", "ON"])[do_swt]))
        logging.info("computing Wavelets from pypwt")
        W = Wavelets(self.data, wname, levels, do_swt=do_swt)
        t0 = time()
        W.forward()
        logging.info("Wavelets.forward() took %.3f ms" % elapsed_ms(t0))
        logging.info("computing Wavelets from pywt")
        t0 = time()
        if not(do_swt):
            Wpy = pywt.wavedec2(self.data, wname, mode="per", level=levels)
        else:
            Wpy = pywt.swt2(self.data, wname, levels)
        logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        self.compare_coeffs(W, Wpy, swt=bool(do_swt))








def test_suite_all_wavelets():
    print("Testing all the %d available filters" % len(available_filters))
    testSuite = unittest.TestSuite()
    maxlev = 2 # beware of filter size reducing the possible number of levels
    for wname in available_filters:
        testSuite.addTest(ParametrizedTestCase.parametrize(TestWavelet, param=(wname, 2, 0)))
    return testSuite


def test_suite_wavelets():
    testSuite = unittest.TestSuite()
    #~ testSuite.addTest(ParametrizedTestCase.parametrize(TestWavelet, param=("haar", 8, 0)))
    testSuite.addTest(ParametrizedTestCase.parametrize(TestWavelet, param=("rbior2.4", 3, 0)))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_wavelets()
    #~ mysuite = test_suite_all_wavelets()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)




