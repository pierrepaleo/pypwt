#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO:
#  - "pywt" vs "PyWavelets"
#    - [OK] Legacy: support pywt
#    - Take PyWavelets new swt order into account

import sys
import unittest
import logging
import numpy as np
from time import time
from copy import deepcopy
from testutils import available_filters, scipy_img

try:
    import pywt
except ImportError:
    print("ERROR : could not find the python module pywt")
    sys.exit(1)
try:
    from pypwt import Wavelets
except ImportError:
    print("ERROR: could not load pypwt. Make sure it is installed (python setup.py install --user)")
    sys.exit(1)
# Version <= 0.5 of PyWavelets uses the word "periodization"
# for the dwt extension mode, instead of "per" for nigma/pywt version.
# These are not compatibible for now.
try:
    pywt_ver_full = pywt.version.full_version
    v = pywt_ver_full.split(".")
    pywt_ver = float(v[0]) + 10**-(len(v[1]))*float(v[1])
    per_kw = "periodization"
except AttributeError: # nigma/pywt
    per_kw = "per"
    pywt_ver = -1.0
    pywt_ver_full = "?"



# Uncomment for odd-size data
# scipy_img = scipy_img[:511, :509]







# Logging
logging.basicConfig(filename='results.log', filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S', level=logging.DEBUG)
# -----



def elapsed_ms(t0):
    return (time()-t0)*1e3


def _calc_errors(arr1, arr2, string=None):
    if string is None: string = ""
    maxerr = np.max(np.abs(arr1 - arr2))
    msg = str("%s max error: %e" % (string, maxerr))
    logging.info(msg)
    return maxerr


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


class TestWavelet(ParametrizedTestCase):


    def setUp(self):
        """
        Set up the TestWavelet class with default parameters.
        """
        # Maximum acceptable error wrt pywt for float32 precision.
        # As the transform are not scaled, the error increases with
        # the number of levels. Thus, self.tol is multiplied with 2**levels
        self.tol = 3e-4
        self.data = scipy_img
        self.do_pywt = False # use pywt when testing reconstruction (for benchmarking)

        # Default arguments when testing only one wavelet
        self.wname = "haar"
        self.levels = 8

        # Bind names to methods
        self.tests = {
            "dwt2": self.dwt2,
            "idwt2": self.idwt2,
            "swt2": self.swt2,
            "iswt2": self.iswt2,
            "dwt": self.dwt,
            "idwt": self.idwt,
            "swt": self.swt,
            "iswt": self.iswt,
        }
        self.W = None

        v_str = str("Using pypwt version %s and pywavelets version %s" % (Wavelets.version(), str(pywt_ver_full)))
        logging.info(v_str)


    def tearDown(self):
        #
        # This one is very important when running many tests.
        # For some reason, it looks like the reference to the Wavelets object
        # is not always updated, causing some errors.
        # This should be investigated further...
        #
        if self.W is not None:
            del self.W


    def test_wavelet(self):
        """
        Method which is actually executed when the test in launched.
        An additional parameter can be passed: param=..., where:
            param[0]: wavelet name
            param[1]: number of levels
            param[2]: input data
            param[3]: what to do ("dwt2", "idwt2", "swt2", "iswt2", "dwt", "idwt", "swt", "iswt"
            param[4]: separable mode for pypwt (default is True)
            param[5]: test-dependent extra parameters
        """

        if self.param is None:
            self.what = "dwt2"
            self.separable = 1
            self.extra_args = None
        else:
            self.wname = self.param["wname"]
            self.levels = self.param["levels"]
            self.data = self.param["data"]
            self.what = self.param["what"]
            self.separable = self.param["separable"]
            self.extra_args = None
            if "extra" in self.param.keys():
                self.extra_args = self.param["extra"]
                if "tol" in self.extra_args.keys(): self.tol = self.extra_args["tol"]
                if "do_pywt" in self.extra_args.keys() and bool(self.extra_args["do_pywt"]): self.do_pywt = True

        # batched WT is not available in nigma/pywt
        if pywt_ver < 0 and self.what in ["dwt", "idwt", "swt", "iswt"] and self.data.ndim > 1:
            self.skipTest("Skipping %s as batch is not supported in this version of pywt" % self.what)

        #FIXME: coefficients in PyWavelets are re-computed, which can lead to significant errors for coif5
        if ("i" not in self.what) and (pywt_ver > 0 and pywt_ver <= 0.5) and (self.wname == "coif5"):
            self.skipTest("Skipping coif5 test for PyWavelets %s" %pywt_ver_full)
        #FIXME: big error for rbio3.1
        if "i" in self.what and self.wname == "rbio3.1":
            self.skipTest("Skipping rbio3.1 inversion")
        # FIXME: big error for bior3.1
        if "i" not in self.what and "swt" in self.what and self.wname == "bior3.1":
            self.skipTest("Skipping bior3.1 SWT")



        # Force an appropriate value for levels
        self.levels = min(self.levels, int(np.log2(min(self.data.shape)//pywt.Wavelet(self.wname).dec_len)))

        # Build the pywt Wavelet instance
        if "swt" in self.what: do_swt = 1
        else: do_swt = 0
        if "2" in self.what: ndim = 2
        else: ndim = 1
        self.W = Wavelets(self.data, self.wname, self.levels, do_separable=self.separable, do_swt=do_swt, ndim=ndim)

        # pywt/PyWavelets do not support odd-size SWT
        if "swt" in self.what:
            nope = 0
            if (self.data.shape[0] & 1): nope = 1
            if (self.W.ndim > 1):
                if (self.data.shape[1] & 1): nope = 1
            if nope:
                if "i" in self.what: # Inversion test can be performed without pywt
                    if self.extra_args is None: self.extra_args = {}
                    self.extra_args["do_pywt"] = False
                else: # Forward transform must be run with pywt for comparisons
                     self.skipTest("Skipping %s with data shape %s, as pywt does not support odd-sized SWT" % (self.what, str(self.data.shape)))

        # Run the test
        if self.what not in self.tests:
            raise ValueError("Unknown test %s" % self.what)
        isbatched = "batched" if self.W.batched1d else ""
        logging.info("Testing %s %s%s with %s, %d levels" % (isbatched, self.what, self.data.shape, self.wname, self.levels))
        self.tests[self.what]()


    def dwt2(self):
        """
        Test pypwt against pywt for DWT2 (wavedec2).
        """
        W = self.W
        levels = self.levels
        wname = self.wname

        # Forward DWT2 with pypwt
        logging.info("computing Wavelets from pypwt")
        t0 = time()
        W.forward()
        logging.info("Wavelets.forward took %.3f ms" % elapsed_ms(t0))

        # Forward DWT2 with pywt
        logging.info("computing wavedec2 from pywt")
        Wpy = pywt.wavedec2(self.data, self.wname, mode=per_kw, level=self.levels)
        logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Compare results
        # FIXME: Error increases when levels increase, since output is scaled.
        tol = self.tol * 2**levels

        W_coeffs = W.coeffs
        if (levels != W.levels):
            err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
            logging.error(err_msg)
            raise ValueError(err_msg)
        A = Wpy[0]
        maxerr = _calc_errors(A, W_coeffs[0], "[app]")
        self.assertTrue(maxerr < tol, msg="[%s] something wrong with the approximation coefficients (%d levels) (errmax = %e)" % (wname, levels, maxerr))
        for i in range(levels): # wavedec2 format
            # FIXME: Error increases when levels increase, since output is scaled.
            tol = self.tol * 2**(i+1)
            D1, D2, D3 = Wpy[levels-i][0], Wpy[levels-i][1], Wpy[levels-i][2]
            logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
            maxerr = _calc_errors(D1, W_coeffs[i+1][0], "[det.H]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 1 at level %d (errmax = %e)" % (wname, i+1, maxerr))
            maxerr = _calc_errors(D2, W_coeffs[i+1][1], "[det.V]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 2 at level %d (errmax = %e)" % (wname, i+1, maxerr))
            maxerr = _calc_errors(D3, W_coeffs[i+1][2], "[det.D]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 3 at level %d (errmax = %e)" % (wname, i+1, maxerr))


    def idwt2(self):
        """
        Test pypwt for DWT reconstruction (waverec2).
        """

        W = self.W
        levels = self.levels
        # inverse DWT with pypwt
        W.forward()
        logging.info("computing Wavelets.inverse from pypwt")
        t0 = time()
        W.inverse()
        logging.info("Wavelets.inverse took %.3f ms" % elapsed_ms(t0))

        if self.do_pywt:
            # inverse DWT with pywt
            Wpy = pywt.wavedec2(self.data, self.wname, mode=per_kw, level=levels)
            logging.info("computing waverec2 from pywt")
            _ = pywt.waverec2(Wpy, self.wname, mode=per_kw)
            logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Check reconstruction
        W_image = W.image
        maxerr = _calc_errors(self.data, W_image, "[rec]")
        self.assertTrue(maxerr < self.tol, msg="[%s] something wrong with the reconstruction (errmax = %e)" % (self.wname, maxerr))


    def swt2(self):
        """
        Test pypwt against pywt for SWT2.
        """
        W = self.W
        levels = self.levels
        wname = self.wname

        # Forward SWT2 with pypwt
        logging.info("computing Wavelets from pypwt")
        t0 = time()
        W.forward()
        logging.info("Wavelets.forward took %.3f ms" % elapsed_ms(t0))

        # Forward SWT2 with pywt
        logging.info("computing wavedec2 from pywt")
        Wpy = pywt.swt2(self.data, self.wname, level=levels)
        logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Compare results
        # FIXME: Error increases when levels increase, since output is scaled.
        tol = self.tol * 2**levels

        W_coeffs = W.coeffs
        if (levels != W.levels):
            err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
            logging.error(err_msg)
            raise ValueError(err_msg)

        # For now pypwt only returns the last appcoeff
        A = Wpy[levels-1][0]
        maxerr = _calc_errors(A, W_coeffs[0], "[app]")
        self.assertTrue(maxerr < tol, msg="[%s] something wrong with the approximation coefficients (%d levels) (errmax = %e)" % (wname, levels, maxerr))
        for i in range(levels): # wavedec2 format. TODO: pywavelets > 0.5 will use another order
            tol = self.tol * 2**(i+1)
            A, D1, D2, D3 = Wpy[i][0], Wpy[i][1][0], Wpy[i][1][1], Wpy[i][1][2]
            logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
            maxerr = _calc_errors(D1, W_coeffs[i+1][0], "[det.H]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 1 at level %d (errmax = %e)" % (wname, i+1, maxerr))
            maxerr = _calc_errors(D2, W_coeffs[i+1][1], "[det.V]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 2 at level %d (errmax = %e)" % (wname, i+1, maxerr))
            maxerr = _calc_errors(D3, W_coeffs[i+1][2], "[det.D]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients 3 at level %d (errmax = %e)" % (wname, i+1, maxerr))


    def iswt2(self):
        """
        Test pypwt for DWT2 reconstruction (iswt2).
        """

        W = self.W
        # inverse DWT with pypwt
        W.forward()
        logging.info("computing Wavelets.inverse from pypwt")
        t0 = time()
        W.inverse()
        logging.info("Wavelets.inverse took %.3f ms" % elapsed_ms(t0))

        if self.do_pywt:
            # inverse DWT with pywt
            Wpy = pywt.swt2(self.data, self.wname, level=self.levels)
            logging.info("computing iswt2 from pywt")
            _ = pywt.iswt2(Wpy, self.wname)
            logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Check reconstruction
        W_image = W.image
        maxerr = _calc_errors(self.data, W_image, "[rec]")
        self.assertTrue(maxerr < self.tol, msg="[%s] something wrong with the reconstruction (errmax = %e)" % (self.wname, maxerr))


    def dwt(self):
        """
        Test pypwt against pywt for DWT (wavedec).
        """
        W = self.W
        levels = self.levels
        wname = self.wname

        # Forward DWT with pypwt
        logging.info("computing Wavelets from pypwt")
        t0 = time()
        W.forward()
        logging.info("Wavelets.forward took %.3f ms" % elapsed_ms(t0))

        # Forward DWT2 with pywt
        logging.info("computing wavedec from pywt")
        Wpy = pywt.wavedec(self.data, self.wname, mode=per_kw, level=levels)
        logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Compare results
        # FIXME: Error increases when levels increase, since output is scaled.
        tol = self.tol * 2**levels

        W_coeffs = W.coeffs
        if (levels != W.levels):
            err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
            logging.error(err_msg)
            raise ValueError(err_msg)
        A = Wpy[0]
        maxerr = _calc_errors(A, W_coeffs[0], "[app]")
        self.assertTrue(maxerr < tol, msg="[%s] something wrong with the approximation coefficients (%d levels) (errmax = %e)" % (wname, levels, maxerr))
        for i in range(levels): # wavedec2 format
            # FIXME: Error increases when levels increase, since output is scaled.
            tol = self.tol * 2**(i+1)
            D1 = Wpy[levels-i]
            logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
            maxerr = _calc_errors(D1, W_coeffs[i+1], "[det]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients at level %d (errmax = %e)" % (wname, i+1, maxerr))


    def idwt(self):
        """
        Test pypwt for DWT reconstruction (waverec).
        """

        W = self.W
        # inverse DWT with pypwt
        W.forward()
        logging.info("computing Wavelets.inverse from pypwt")
        t0 = time()
        W.inverse()
        logging.info("Wavelets.inverse took %.3f ms" % elapsed_ms(t0))

        if self.do_pywt:
            # inverse DWT with pywt
            Wpy = pywt.wavedec(self.data, self.wname, mode=per_kw, level=self.levels)
            logging.info("computing waverec from pywt")
            _ = pywt.waverec(Wpy, self.wname, mode=per_kw)
            logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Check reconstruction
        W_image = W.image
        maxerr = _calc_errors(self.data, W_image, "[rec]")
        self.assertTrue(maxerr < self.tol, msg="[%s] something wrong with the reconstruction (errmax = %e)" % (self.wname, maxerr))


    def swt(self):
        """
        Test pypwt against pywt for SWT.
        """
        W = self.W
        levels = self.levels
        wname = self.wname

        # Forward DWT with pypwt
        logging.info("computing Wavelets from pypwt")
        t0 = time()
        W.forward()
        logging.info("Wavelets.forward took %.3f ms" % elapsed_ms(t0))

        # Forward DWT2 with pywt
        logging.info("computing swt from pywt")
        Wpy = pywt.swt(self.data, self.wname, level=levels)
        logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Compare results
        # FIXME: Error increases when levels increase, since output is scaled.
        tol = self.tol * 2**levels

        W_coeffs = W.coeffs
        if (levels != W.levels):
            err_msg = str("compare_coeffs(): pypwt instance has %d levels while pywt instance has %d levels" % (W.levels, levels))
            logging.error(err_msg)
            raise ValueError(err_msg)
        A = Wpy[0][0]
        W_a = W_coeffs[0] if W.batched1d else W_coeffs[0].ravel()
        maxerr = _calc_errors(A, W_a, "[app]") #
        self.assertTrue(maxerr < tol, msg="[%s] something wrong with the approximation coefficients (%d levels) (errmax = %e)" % (wname, levels, maxerr))
        for i in range(levels): # wavedec2 format
            # FIXME: Error increases when levels increase, since output is scaled.
            tol = self.tol * 2**(i+1)
            D1 = Wpy[levels-i-1][1] # TODO: take the new PyWavelet swt order into account
            logging.info("%s Level %d %s" % ("-"*5, i+1, "-"*5))
            W_D1 = W_coeffs[i+1] if W.batched1d else W_coeffs[i+1].ravel()
            maxerr = _calc_errors(D1, W_D1, "[det]")
            self.assertTrue(maxerr < tol, msg="[%s] something wrong with the detail coefficients at level %d (errmax = %e)" % (wname, i+1, maxerr))


    def iswt(self):
        """
        Test pypwt for ISWT reconstruction.
        """

        W = self.W
        # inverse DWT with pypwt
        W.forward()
        logging.info("computing Wavelets.inverse from pypwt")
        t0 = time()
        W.inverse()
        logging.info("Wavelets.inverse took %.3f ms" % elapsed_ms(t0))

        # PyWavelets <= 0.5 does not have an "axis" property for iswt
        #~ if self.do_pywt:
            #~ # inverse DWT with pywt
            #~ Wpy = pywt.swt(self.data, self.wname, level=self.levels)
            #~ logging.info("computing waverec from pywt")
            #~ _ = pywt.iswt(Wpy, self.wname)
            #~ logging.info("pywt took %.3f ms" % elapsed_ms(t0))

        # Check reconstruction
        W_image = W.image
        maxerr = _calc_errors(self.data, W_image, "[rec]")
        self.assertTrue(maxerr < self.tol, msg="[%s] something wrong with the reconstruction (errmax = %e)" % (self.wname, maxerr))




# End of class
# ----------------





def test_wavelet(what, data=None, levels=None, wname=None):
    """
    Entry point for running a test.
    It returns a unittest.TestSuite() instance.

    Parameters
    -----------
    what: str
        Name of the test
    data: numpy.ndarray, optional
        Input data
    levels: int, optional
        Number of decomposition levels
    wname: str, optional
        Name of the wavelet to test. If no argument is provided,
        all wavelets will be tested.

    Example
    ---------
    runner = unittest.TextTestRunner()
    mysuite = test_wavelet("dwt2", my_img, 8)
    runner.run(mysuite)
    """

    if data is None: data = scipy_img
    if levels is None: levels = 999 # Max level will be automatically computed

    testSuite = unittest.TestSuite()
    if what == "dwt2":
        par = {
            "levels": levels,
            "data": data,
            "what": "dwt2",
            "separable": 1,
            "extra": {
                "tol": 4e-4, # bior3.1....
            }
        }
    if what == "idwt2":
        par = {
            "levels": levels,
            "data": data,
            "what": "idwt2",
            "separable": 1,
            "extra": {
                "tol": 7e-4,
                "do_pywt": True # False # set to True for benchmarking - can be slow !
            }
        }
    if what == "swt2":
        par = {
            "levels": levels,
            "data": data,
            "what": "swt2",
            "separable": 1,
            "extra": {
                "tol": 3e-4,
            }
        }
    if what == "iswt2":
        par = {
            "levels": levels,
            "data": data,
            "what": "iswt2",
            "separable": 1,
            "extra": {
                "tol": 4e-4,
                "do_pywt": True # False # set to True for benchmarking - can be slow !
            }
        }
    if what == "dwt":
        par = {
            "levels": levels,
            "data": data[50, :],
            "what": "dwt",
            "separable": 1,
            "extra": {
                "tol": 1e-4,
            }
        }
    if what == "dwt_batched":
        par = {
            "levels": levels,
            "data": data,
            "what": "dwt",
            "separable": 1,
            "extra": {
                "tol": 1e-4,
            }
        }
    if what == "idwt":
        par = {
            "levels": levels,
            "data": data[50, :],
            "what": "idwt",
            "separable": 1,
            "extra": {
                "tol": 2e-4,
                "do_pywt": False # set to True for benchmarking
            }
        }
    if what == "idwt_batched":
        par = {
            "levels": levels,
            "data": data,
            "what": "idwt",
            "separable": 1,
            "extra": {
                "tol": 5e-4, # bior* wavelets...
                "do_pywt": False # set to True for benchmarking
            }
        }

    if what == "swt":
        par = {
            "levels": levels,
            "data": data[50, :],
            "what": "swt",
            "separable": 1,
            "extra": {
                "tol": 4e-5,
            }
        }
    if what == "swt_batched":
        par = {
            "levels": levels,
            "data": data,
            "what": "swt",
            "separable": 1,
            "extra": {
                "tol": 1.1e-4, # db17
            }
        }
    if what == "iswt":
        par = {
            "levels": levels,
            "data": data[50, :],
            "what": "iswt",
            "separable": 1,
            "extra": {
                "tol": 1e-4,
                "do_pywt": False # set to True for benchmarking
            }
        }
    if what == "iswt_batched":
        par = {
            "levels": levels,
            "data": data,
            "what": "iswt",
            "separable": 1,
            "extra": {
                "tol": 3e-4, # bior* wavelets...
                "do_pywt": False # set to True for benchmarking - can be slow !
            }
        }
    if wname is None: wnames = available_filters
    else: wnames = [wname] # TODO: iterable user-provided "wname"
    for wname in wnames:
        par_curr = deepcopy(par) # otherwise the same dict is used for all tests
        par_curr["wname"] = wname
        testcase = ParametrizedTestCase.parametrize(TestWavelet, param=par_curr)
        testSuite.addTest(testcase)
    return testSuite











def test_all():
    suite = unittest.TestSuite()
    suite.addTest(test_wavelet("dwt2"))
    suite.addTest(test_wavelet("idwt2"))
    suite.addTest(test_wavelet("swt2"))
    suite.addTest(test_wavelet("iswt2"))
    suite.addTest(test_wavelet("dwt"))
    suite.addTest(test_wavelet("dwt_batched"))
    suite.addTest(test_wavelet("idwt"))
    suite.addTest(test_wavelet("idwt_batched"))
    suite.addTest(test_wavelet("swt"))
    suite.addTest(test_wavelet("swt_batched"))
    suite.addTest(test_wavelet("iswt"))
    suite.addTest(test_wavelet("iswt_batched"))
    return suite



if __name__ == '__main__':
    mysuite = test_all()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)


