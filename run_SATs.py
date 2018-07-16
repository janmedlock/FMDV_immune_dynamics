#!/usr/bin/python3

import time

import numpy
import pandas

import herd
import run_many


def run_SATs(nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        p = herd.Parameters(SAT=SAT)
        print('Running SAT {}.'.format(SAT))
        t0 = time.time()
        results[SAT] = run_many.run_many(nruns, p, tmax, *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['SAT'])


if __name__ == '__main__':
    nruns = 2
    tmax = 1
    debug = False
    export_data = True

    data = run_SATs(nruns, tmax, debug=debug)

    if export_data:
        data.to_pickle('run_SATs.pkl')
