#!/usr/bin/python3

import numpy
import pandas

import herd
from run_start_times import run_start_times


def run_start_times_SATs(nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        results[SAT] = run_start_times(nruns, SAT, tmax, *args, **kwargs)
    return pandas.concat(results, names=['SAT'])


if __name__ == '__main__':
    nruns = 10000
    tmax = numpy.inf

    data = run_start_times_SATs(nruns, tmax)
    data.to_pickle('run_start_times_SATs.pkl')
