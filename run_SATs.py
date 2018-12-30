#!/usr/bin/python3

import os.path
import time

import pandas

import herd
from run_many import run_many


def run_SATs(chronic, nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        p = herd.Parameters(SAT=SAT, chronic=chronic)
        print('Running SAT {}.'.format(SAT))
        t0 = time.time()
        results[SAT] = run_many(nruns, p, tmax, *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['SAT'], copy=False)


if __name__ == '__main__':
    chronic = True
    nruns = 1000
    tmax = 10

    data = run_SATs(chronic, nruns, tmax)
    _filebase, _ = os.path.splitext(__file__)
    if chronic:
        _filebase += '_chronic'
    _picklefile = _filebase + '.pkl'
    data.to_pickle(_picklefile)
