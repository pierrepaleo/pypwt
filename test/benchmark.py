#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import numpy as np
import pywt
from pypwt import Wavelets
from bench import Benchmark
from testutils import scipy_img, create_data_to_good_size, what_to_params

try:
    pywt_ver_full = pywt.version.full_version
    v = pywt_ver_full.split(".")
    pywt_ver = float(v[0]) + 10**-(len(v[1]))*float(v[1])
    per_kw = "periodization"
except AttributeError: # nigma/pywt
    per_kw = "per"
    pywt_ver = -1.0
    pywt_ver_full = "?"


# You can customize the following
# -----------------------------------
data = scipy_img
what = "dwt2"
Wname = ["haar", "db20"] # can be a list of <= 3 elements
levels = 999 # cliped to max level

data_sizes = [
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    #(4096, 4096),
]
# -----------------------------------





if type(Wname) is str: Wname = [Wname]
bench = Benchmark()
#thetitle = str("%s (%s)" % (what_to_params[what]["name"], wname))
thetitle = str("%s" % what_to_params[what]["name"])
bench.new_figure(thetitle, xlabel="Number of points", ylabel="Time (ms)", xlog=True, ylog=True, xlims=(1.1e-2, 1), ylims=(1e-2, 1))
markers = ["o--", "o-."]
for i,wname in enumerate(Wname):
    bench.new_curve("pywt: " + wname, marker=markers[i])
    bench.new_curve("PDWT: " + wname, marker=markers[i])

leg = bench.legend()
leg.draggable()

results_pywt = []
results_pypwt = []


# pywt does not need to compute a plan for each image size
def W_pywt_exec(wname, lev):
    if "s" in what: # pywt always use the periodic boundary condition with swt
        if "batched" in what: # pyPwt makes transforms along contiguous dimension
            return lambda x : what_to_params[what]["pywt_function"](x, wname, level=lev, axis=1)
        else:
            return lambda x : what_to_params[what]["pywt_function"](x, wname, level=lev)
    else:
        if "batched" in what:
            return lambda x : what_to_params[what]["pywt_function"](x, wname, mode=per_kw, level=lev, axis=1)
        else:
            return lambda x : what_to_params[what]["pywt_function"](x, wname, mode=per_kw, level=lev)


for wname in Wname:
    for size in data_sizes:
        data_curr = create_data_to_good_size(data, size)
        # Make sure to use contiguous array for benchmarking
        if not(data_curr.flags["C_CONTIGUOUS"]): data_curr = np.ascontiguousarray(data_curr)
        #data_curr = data_curr.astype(np.float32)

        # pyPwt needs to compute a plan for each image size
        do_swt = what_to_params[what]["do_swt"]
        ndim = what_to_params[what]["ndim"]
        print("creating Wavelets()")
        W_pypwt = Wavelets(data_curr, wname=wname, levels=levels, do_swt=do_swt, ndim=ndim)
        print("OK")
        print(W_pypwt)
        raw_input()
        lev = W_pypwt.levels
        # Additionally, for inversion:
        if "i" in what:
            W_pypwt.forward()

        def W_pypwt_exec():
            if "i" not in what:
                print("Doing forward() size %s " % str(size))
                W_pypwt.forward()
                print("... OK")
            else:
                W_pypwt.inverse()

        def W_pypwt_exec_with_copy(x):
            # Takes the H<->D transfers into account for the benchmark
            W_pywt.set_image(x) # Here works for forward...
            W_pywt.forward()
            c = W_pywt.coeffs

        xval = np.prod(size)/1e6
        label = str(size)
        res_pywt = bench.add_bench_result("pywt: " + wname, xval, W_pywt_exec(wname, lev), label=label, command_args=data_curr, verbose=True, nexec=1)
        res_pypwt = bench.add_bench_result("PDWT: " + wname, xval, W_pypwt_exec, label=label, verbose=True, nexec=1)
        results_pywt.append(res_pywt)
        results_pypwt.append(res_pypwt)
        print("end of size %s" % str(size))
        del W_pypwt

bench.fit_plots_to_fig(margin_x=0.2, margin_y=0.2)
print(np.array(results_pywt)/np.array(results_pypwt))
raw_input()






