#!/usr/bin/env python
from __future__ import print_function
import sys
from testutils import what_to_params

def parse_result(fname, fname_out):
    fid_o = open(fname_out, "a")
    with open(fname) as fid:
        flag_begin = 0
        for line in fid:
            if "INFO" not in line: continue
            s = line.split("INFO ")[1]

            # Extracts "Testing  swt2(512, 512) with haar, 5 levels"
            if "Testing" in s:
                L = s.split("Testing")[1].split("(")
                what = L[0].lstrip()
                size = "(" + L[1].split(")")[0] + ")"
                wname = L[1].split()[3].rstrip(",")
                levels = L[1].split()[4]
                if flag_begin == 0:
                    print(r" \textbf{%s} & PyWavelets (ms) & PDWT (ms) & Speed-up \\" % what.upper(), file=fid_o)
                    print(r"\hline", file=fid_o)
                    flag_begin = 1

            if "Wavelets" in s and "took" in s:
                time_pypwt = float(s.split()[2])
            if "pywt took" in s:
                time_pywt = float(s.split()[2])
                # pywt comes after pypwt in the test
                speedup = time_pywt/time_pypwt
                print(r"  %s  &  %.2f  & %.2f  & %.0f \\" %(size, time_pywt, time_pypwt, speedup), file=fid_o)
                print(r"\hline", file=fid_o)

    fid_o.close()
    return wname, what


def table_header(fname):
    with open(fname, 'a') as fid_o:
        print(r"\begin{table}[H]", file=fid_o)
        print(r"\begin{center}\begin{tabular}{|c|c|c|c|}", file=fid_o)
        print(r"\hline", file=fid_o)
        #print(r" & PyWavelets (ms) & PDWT (ms) & Speed-up \\", file=fid_o)
        #print(r"\hline", file=fid_o)



def table_footer(fname, wname, what):
    with open(fname, 'a') as fid_o:
        print(r"\end{tabular}\end{center}", file=fid_o)
        print(r'\caption{Results of %s with the ``%s" wavelets}' % (what_to_params[what]["name"], wname), file=fid_o)
        print(r"\end{table}", file=fid_o)




if __name__ == "__main__":
    nargs = len(sys.argv[1:])
    if nargs == 0:
        fname = "results.log"
        fname_out = "results.tex"
    else:
        fname = sys.argv[1]
        if nargs == 2: fname_out = sys.argv[2]
        else: fname_out = "results.tex"

    table_header(fname_out)
    wname, what = parse_result(fname, fname_out)
    table_footer(fname_out, wname, what)
