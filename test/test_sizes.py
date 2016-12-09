#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of ESRF nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE


import numpy as np
import unittest
from testutils import scipy_img, create_data_to_good_size
from test_wavelets import test_wavelet








def run():

    # You can customize the following
    # -----------------------------------
    data = scipy_img
    data_sizes = [
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        #(4096, 4096),
    ]
    levels = 999 # will be cliped to max possible level
    wname = "db20"
    what = "dwt"
    # -----------------------------------

    testSuite = unittest.TestSuite()
    for size in data_sizes:
        data_curr = create_data_to_good_size(data, size)
        # Make sure to use contiguous array for benchmarking
        if not(data_curr.flags["C_CONTIGUOUS"]): data_curr = np.ascontiguousarray(data_curr)
        testSuite.addTest(test_wavelet(what, data_curr, levels=levels, wname=wname))

    runner = unittest.TextTestRunner()
    if not runner.run(testSuite).wasSuccessful():
        exit(1)


if __name__ == '__main__':

    run()
