#!/usr/bin/python3

import copy
import time

from joblib import delayed, Parallel
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
    # Combine SAT and `None`.
    s = pandas.concat((samples[SAT], samples[None]), axis=1)
    print('Running SAT {}.'.format(SAT))
    t0 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(i, parameters, sample, tmax, *args, **kwargs)
        for i, sample in s.iterrows())
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, keys=range(len(samples)), names=['sample'])


def run_SATs(tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        results[SAT] = run_samples(SAT, tmax, *args, **kwargs)
    return pandas.concat(results, names=['SAT'])


if __name__ == '__main__':
    tmax = 1

    data = run_SATs(tmax)
    data.to_pickle('run_samples.pkl')
