#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2016 Pierre Paleo <pierre.paleo@esrf.fr>
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

try:
    import matplotlib
    matplotlib.use('tkagg') # ion() with qt is more complicated
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError("matploblib must be installed in order to run benchmark.")
import numpy as np
from time import time

# Use better color maps
_have_brewer = False
try:
    import brewer2mpl
    _have_brewer = True
except ImportError:
    _have_brewer = False




class Benchmark(object):

    def __init__(self): # One "figure" (window) per benchmark
        self.figname = None
        self.fig = None
        self.ax = None
        self.xdata = []
        self.ydata = []
        self.plots = {}
        self.lims = {}
        self.curvenames = []
        self.xlabels = {}

        self.was_interactive = plt.isinteractive()
        plt.ion()


    def new_figure(self, figname, xlims=None, ylims=None, xlabel=None, ylabel=None, xlog=False, ylog=False):
        self.figname = figname
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        xmin, xmax = xlims if xlims else (0., 1.)
        ymin, ymax = ylims if ylims else (0., 1.)
        self.lims = (xmin, xmax, ymin, ymax)
        self.ax.set_title(figname)
        if xlabel: self.ax.set_xlabel(xlabel)
        if ylabel: self.ax.set_ylabel(ylabel)
        if xlog: self.ax.set_xscale("log")
        if ylog: self.ax.set_yscale("log")
        # Use better colors if brewer2mpl is installed
        global _have_brewer
        if _have_brewer:
            bcolors = brewer2mpl.get_map("RdBu", "Diverging", 6) # TODO: with more classes ?
            plcolors = bcolors.mpl_colors
            Ncol = len(plcolors)
            # Re-order the color cycle
            self.colors = []
            for i in range(Ncol//2):
                    self.colors.append(plcolors[i])
                    self.colors.append(plcolors[Ncol-1-i])
            self.ax.set_color_cycle(self.colors)
        plt.show()


    def new_curve(self, name, xdata=None, ydata=None, marker="o-", lw=2):
        if xdata:
            if not(ydata): raise ValueError("Please provide ydata if providing xdata")
            xdata = list(xdata)
            ydata = list(ydata)
        else:
            xdata, ydata = [], []

        icol = len(self.curvenames)
        self.plots[name] = self.ax.plot(xdata, ydata, marker, lw=lw)[0]
        self.curvenames.append(name)
        if xdata: self.update_fig_lims(self.plots[name], xdata, ydata)



    def add_point(self, name, X, Y, label=None):
        dx, dy = self.plots[name].get_data()
         # TODO : temporary workaround ; optimize !
        dx = list(dx)
        dy = list(dy)
        dx.append(X)
        dy.append(Y)
        self.plots[name].set_data(dx, dy)
        self.update_fig_lims(name, dx, dy)

        if label: # TODO: improve
            if name in self.xlabels: self.xlabels[name].append(label)
            else: self.xlabels[name] = [label]
            self.ax.set_xticks(dx)
            self.ax.set_xticklabels(self.xlabels[name], size="small")
        self.fig.canvas.draw()


    def add_bench_result(self, name, xval, command, nexec=3, mode="best", label=None, command_args=None, verbose=False):
        """
        example:

        bench.add_bench_result("mycurve", 2048*2048, plan_fft2.execute)
        """
        if not(callable(command)): raise ValueError("Please provide a callable function")
        results_ms = []
        for i in range(nexec):
            t0 = time()
            if command_args is None: command()
            else: command(command_args) # *command_args
            t = time()
            elapsed_ms = (t-t0)*1e3
            results_ms.append(elapsed_ms)

        if mode == "best": # IPython default
            yval = min(results_ms)
        elif mode == "median":
            yval = np.median(results_ms)
        elif mode == "mean": # bad idea for profiling
            yval = np.mean(results_ms)
        elif mode == "worst":
            yval = np.max(results_ms)
        self.add_point(name, xval, yval, label=label)
        if verbose:
            print("%s \t %s \t %.3f ms" % (name, label, yval))
        return yval



    def update_fig_lims(self, theplot, xdata, ydata):
        cxmin, cxmax, cymin, cymax = self.lims
        cxmin = min(cxmin, min(xdata))
        cxmax = max(cxmax, max(xdata))
        cymin = min(cymin, min(ydata))
        cymax = max(cymax, max(ydata))
        self.lims = (cxmin, cxmax, cymin, cymax)
        self.ax.set_xlim([cxmin, cxmax])
        self.ax.set_ylim([cymin, cymax])


    def fit_plots_to_fig(self, margin_x=0.1, margin_y = 0.1):
        """
        Fit the data plots to the figure axis.
        margin_x and margin_y specify the margin to the figure borders.
        """
        min_x = np.inf
        max_x = -np.inf
        min_y = np.inf
        max_y = -np.inf
        for plot_name in self.plots.keys():
            X, Y = self.plots[plot_name].get_data()
            Xa = np.array(X)
            Ya = np.array(Y)
            xmin, xmax = Xa.min(), Xa.max()
            ymin, ymax = Ya.min(), Ya.max()
            if xmin < min_x: min_x = xmin
            if ymin < min_y: min_y = ymin
            if xmax > max_x: max_x = xmax
            if ymax > max_y: max_y = ymax

        min_x = min_x - np.abs(min_x)*margin_x
        max_x = max_x + np.abs(max_x)*margin_x
        min_y = min_y - np.abs(min_y)*margin_y
        max_y = max_y + np.abs(max_y)*margin_y
        self.ax.set_xlim([min_x, max_x])
        self.ax.set_ylim([min_y, max_y])
        self.fig.canvas.draw()


    def legend(self):
        return self.ax.legend(self.curvenames)


    def __del__(self):
        if not(self.was_interactive): plt.ioff()



