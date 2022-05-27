#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples.h5`.'''


import copy
import itertools
import os

from joblib import delayed, Parallel
import numpy
import pandas

import h5
import herd
import herd.samples
import run


_path = os.path.join(os.path.dirname(__file__),
                     'samples')
_t_name = 'time (y)'


def run_one(parameters, sample, tmax, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for (k, v) in sample.items():
        setattr(p, k, v)
    return run.run_one(p, tmax, sample_number, *args, **kwargs)


def run_one_and_save(parameters, sample, tmax, sample_number, path,
                     *args, **kwargs):
    '''Run one simulation.'''
    filename = os.path.join(path, f'{sample_number}.npy')
    if not os.path.exists(filename):
        dfr = run_one(parameters, sample, tmax, sample_number,
                      *args, **kwargs)
        # Save the data for this sample.
        numpy.save(filename, dfr.to_records())


def _get_jobs_SAT(SAT, samples, tmax, path):
    '''Get jobs to run in parallel for one SAT.'''
    path_SAT = os.path.join(path, str(SAT))
    os.makedirs(path_SAT, exist_ok=True)
    p = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    return (delayed(run_one_and_save)(p, s, tmax, n, path_SAT,
                                      logging_prefix=logging_prefix)
            for (n, s) in samples.iterrows())


def run_samples(tmax, n_jobs=-1):
    samples = herd.samples.load()
    os.makedirs(_path, exist_ok=True)
    jobs = itertools.chain.from_iterable(
        _get_jobs_SAT(SAT, samples[SAT], tmax, _path)
        for SAT in run._SATs)
    Parallel(n_jobs=n_jobs)(jobs)


def _get_sample_number(filename):
    base, _ = os.path.splitext(filename)
    return int(base)


def combine():
    with h5.HDFStore('samples.h5', mode='a') as store:
        # (SAT, sample) that are already in `store`.
        store_idx = store.get_index().droplevel(_t_name).unique()
        for SAT in map(int, sorted(os.listdir(_path))):
            path_SAT = os.path.join(_path, str(SAT))
            # Sort in integer order.
            for filename in sorted(os.listdir(path_SAT),
                                   key=_get_sample_number):
                sample = _get_sample_number(filename)
                if (SAT, sample) not in store_idx:
                    path_sample = os.path.join(path_SAT, filename)
                    recarray = numpy.load(path_sample)
                    df = pandas.DataFrame.from_records(recarray,
                                                       index=_t_name)
                    run._prepend_index_levels(df,
                                              SAT=SAT,
                                              sample=sample)
                    print('Inserting '
                          + ', '.join((f'SAT={SAT}',
                                       f'sample={sample}'))
                          + '.')
                    store.put(df, min_itemsize=run._min_itemsize)
                    # os.remove(path_sample)


if __name__ == '__main__':
    tmax = 10

    run_samples(tmax)
    # combine()
