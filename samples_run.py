#!/usr/bin/python3

import copy
import itertools
import os

from joblib import delayed, Parallel
import numpy

import herd
import herd.samples
import run


_path = 'samples'
_index = 'time (y)'


def _run_sample(parameters, sample, tmax, path, sample_number, logging_prefix):
    '''Run one simulation.'''
    filename = os.path.join(path, f'{sample_number}.npy')
    if not os.path.exists(filename):
        p = copy.copy(parameters)
        for (k, v) in sample.items():
            setattr(p, k, v)
        h = herd.Herd(p, run_number=sample_number,
                      logging_prefix=logging_prefix)
        df = h.run(tmax)
        # Save the data for this sample.
        numpy.save(filename, df.to_records())


def _run_samples_SAT(SAT, tmax, path):
    '''Run many simulations in parallel.'''
    path_SAT = os.path.join(path, str(SAT))
    os.makedirs(path_SAT, exist_ok=True)
    logging_prefix = f'SAT{SAT}, '
    parameters = herd.Parameters(SAT=SAT)
    samples = herd.samples.load(SAT=SAT)
    return (delayed(_run_sample)(parameters, s, tmax, path_SAT, i,
                                 logging_prefix)
            for (i, s) in samples.iterrows())


def run_samples(tmax):
    os.makedirs(_path, exist_ok=True)
    jobs = itertools.chain.from_iterable(
        _run_samples_SAT(SAT, tmax, _path)
        for SAT in run._SATs)
    Parallel(n_jobs=-1)(jobs)


if __name__ == '__main__':
    tmax = 10

    run_samples(tmax)
