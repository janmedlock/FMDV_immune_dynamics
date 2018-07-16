#!/usr/bin/python3

import time

import numpy
import pandas

import herd
from run_many import run_many


def run_start_times(nruns, SAT, tmax, *args, **kwargs):
    results = {}
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(SAT=SAT)
        p.start_time = start_time
        print('Running SAT {}, start time {}.'.format(SAT, start_time))
        t0 = time.time()
        results[start_time] = run_many(nruns, p, tmax, *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['start_time'])


if __name__ == '__main__':
    nruns = 4
    SAT = 1
    tmax = 1

    data = run_start_times(nruns, SAT, tmax)
    data.to_pickle('run_start_times.pkl')
