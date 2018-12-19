#!/usr/bin/python3

import copy
import os.path
import time

from joblib import delayed, Parallel
import numpy
import pandas

import herd
from herd.samples import samples


def run_one(run_number, parameters, sample, tmax, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for k, v in sample.items():
        setattr(p, k, v)
    h = herd.Herd(p, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def run_samples(SAT, tmax, *args, **kwargs):
    '''Run many simulations in parallel.'''
    parameters = herd.Parameters(SAT=SAT)
    print('Running SAT {}.'.format(SAT))
    t0 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(i, parameters, s, tmax, *args, **kwargs)
        for i, s in samples[SAT].iterrows())
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, keys=range(len(samples)), names=['sample'],
                         copy=False)


def run_SATs(tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        results[SAT] = run_samples(SAT, tmax,
                                   logging_prefix='SAT {}, '.format(SAT),
                                   *args, **kwargs)
    return pandas.concat(results, names=['SAT'], copy=False)


if __name__ == '__main__':
    tmax = 10

    data = run_SATs(tmax)

    _filebase, _ = os.path.splitext(__file__)
    _picklefile = _filebase + '.pkl'
    data.to_pickle(_picklefile)
