#!/usr/bin/python3

import time

import pandas

import herd
from run_many import run_many


def run_SATs(nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        p = herd.Parameters(SAT=SAT)
        print('Running SAT {}.'.format(SAT))
        t0 = time.time()
        results[SAT] = run_many(nruns, p, tmax, *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['SAT'], copy=False)


if __name__ == '__main__':
    nruns = 1000
    tmax = 10

    data = run_SATs(nruns, tmax)
    data.to_pickle('run_SATs.pkl')
