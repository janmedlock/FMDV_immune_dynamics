import copy
import time

from joblib import delayed, Parallel
import numpy
import pandas

import herd
import herd.samples


_SATs = (1, 2, 3)


def run_one(run_number, parameters, tmax, *args, **kwargs):
    '''Run one simulation.'''
    h = herd.Herd(parameters, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def run_many(nruns, parameters, tmax, *args, **kwargs):
    '''Run many simulations in parallel.'''
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(i, parameters, tmax, *args, **kwargs)
        for i in range(nruns))
    # Make 'run' the outer row index.
    return pandas.concat(results, keys=range(nruns), names=['run'],
                         copy=False)


def run_SATs(chronic, nruns, tmax, *args, **kwargs):
    results = {}
    for SAT in _SATs:
        p = herd.Parameters(SAT=SAT, chronic=chronic)
        print('Running SAT {}.'.format(SAT))
        t0 = time.time()
        results[SAT] = run_many(nruns, p, tmax, *args, **kwargs)
        t1 = time.time()
        print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, names=['SAT'], copy=False)


def run_start_times(nruns, SAT, chronic, tmax, logging_prefix='',
                    *args, **kwargs):
    results = {}
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(SAT=SAT, chronic=chronic)
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


def run_start_times_SATs(nruns, chronic, tmax, *args, **kwargs):
    results = {}
    for SAT in _SATs:
        results[SAT] = run_start_times(nruns, SAT, chronic, tmax,
                                       logging_prefix='SAT {}, '.format(SAT),
                                       *args, **kwargs)
    return pandas.concat(results, names=['SAT'], copy=False)


def _run_sample(run_number, parameters, sample, tmax, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for k, v in sample.items():
        setattr(p, k, v)
    h = herd.Herd(p, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def _run_samples_SAT(SAT, chronic, tmax, *args, **kwargs):
    '''Run many simulations in parallel.'''
    samples = herd.samples.load(chronic=chronic)[SAT]
    parameters = herd.Parameters(SAT=SAT, chronic=chronic)
    print('Running SAT {}.'.format(SAT))
    t0 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(_run_sample)(i, parameters, s, tmax, *args, **kwargs)
        for i, s in samples.iterrows())
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))
    return pandas.concat(results, keys=range(len(samples)),
                         names=['sample'], copy=False)


def run_samples(chronic, tmax, *args, **kwargs):
    results = {}
    for SAT in (1, 2, 3):
        results[SAT] = _run_samples_SAT(SAT, chronic, tmax,
                                        logging_prefix='SAT {}, '.format(SAT),
                                        *args, **kwargs)
    return pandas.concat(results, names=['SAT'], copy=False)
