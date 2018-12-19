#!/usr/bin/python3

import os.path

import numpy
import pandas

import herd
from run_start_times import run_start_times


def run_start_times_SATs(nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        results[SAT] = run_start_times(nruns, SAT, tmax,
                                       logging_prefix='SAT {}, '.format(SAT),
                                       *args, **kwargs)
    return pandas.concat(results, names=['SAT'], copy=False)


if __name__ == '__main__':
    nruns = 10000
    tmax = 10

    data = run_start_times_SATs(nruns, tmax)

    _filebase, _ = os.path.splitext(__file__)
    _picklefile = _filebase + '.pkl'
    data.to_pickle(_picklefile)
