#!/usr/bin/python3

import os.path
import time

import numpy
import pandas

import herd
from run_many import run_many


def run_start_times(nruns, SAT, tmax, logging_prefix='', *args, **kwargs):
    results = {}
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(SAT=SAT)
        p.start_time = start_time
        logging_prefix_ = (logging_prefix
                           + 'Start time {:g} / 12'.format(start_time * 12))
        print('Running {}.'.format(logging_prefix_))
        logging_prefix_ += ', '
        t0 = time.time()
        results[start_time] = run_many(nruns, p, tmax,
                                       logging_prefix=logging_prefix_,
                                       *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['start_time'], copy=False)


if __name__ == '__main__':
    nruns = 4
    SAT = 1
    tmax = 1

    data = run_start_times(nruns, SAT, tmax)

    _filebase, _ = os.path.splitext(__file__)
    _picklefile = _filebase + '.pkl'
    data.to_pickle(_picklefile)
