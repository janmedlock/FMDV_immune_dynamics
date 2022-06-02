#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples_run.h5`.'''


import copy
import itertools
import pathlib

from joblib import delayed, Parallel
import numpy
import pandas

import h5
import herd
import herd.samples
import run


store_path = pathlib.Path(__file__).with_suffix('.h5')
samples_path = store_path.with_suffix('')

_t_name = 'time (y)'


def run_one(parameters, sample, tmax, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for (k, v) in sample.items():
        setattr(p, k, v)
    return run.run_one(p, tmax, sample_number, *args, **kwargs)


def run_one_and_save(parameters, sample, tmax, sample_number, path, *args,
                     touch=True, **kwargs):
    '''Run one simulation.'''
    sample_path = path.joinpath(f'{sample_number}.npy')
    if not sample_path.exists():
        if touch:
            sample_path.touch(exist_ok=False)
        dfr = run_one(parameters, sample, tmax, sample_number,
                      *args, **kwargs)
        # Save the data for this sample.
        numpy.save(sample_path, dfr.to_records())


def _get_jobs_SAT(SAT, samples, tmax, path, *args, **kwargs):
    '''Get jobs to run in parallel for one SAT.'''
    SAT_path = path.joinpath(str(SAT))
    SAT_path.mkdir(exist_ok=True)
    p = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    return (delayed(run_one_and_save)(p, s, tmax, n, SAT_path, *args,
                                      logging_prefix=logging_prefix, **kwargs)
            for (n, s) in samples.iterrows())


def run_samples(tmax, *args,
                n_jobs=-1, **kwargs):
    samples = herd.samples.load()
    samples_path.mkdir(exist_ok=True)
    jobs = itertools.chain.from_iterable(
        _get_jobs_SAT(SAT, samples[SAT], tmax, samples_path, *args, **kwargs)
        for SAT in (3, 2, 1))
    Parallel(n_jobs=n_jobs)(jobs)


def _get_SAT(path):
    return int(path.name)


def _get_sample_number(path):
    return int(path.stem)


def combine(unlink=True):
    with h5.HDFStore(store_path, mode='a') as store:
        # (SAT, sample) that are already in `store`.
        store_idx = store.get_index().droplevel(_t_name).unique()
        SAT_paths = sorted(samples_path.iterdir(), key=_get_SAT)
        for SAT_path in SAT_paths:
            SAT = _get_SAT(SAT_path)
            sample_paths = sorted(SAT_path.iterdir(), key=_get_sample_number)
            for sample_path in sample_paths:
                sample = _get_sample_number(sample_path)
                if (SAT, sample) not in store_idx:
                    recarray = numpy.load(sample_path)
                    dfr = pandas.DataFrame.from_records(recarray,
                                                        index=_t_name)
                    run.prepend_index_levels(dfr, SAT=SAT, sample=sample)
                    print('Inserting '
                          + ', '.join((f'SAT={SAT}',
                                       f'sample={sample}'))
                          + '.')
                    store.put(dfr)
                if unlink:
                    sample_path.unlink()
            if unlink:
                SAT_path.rmdir()
        if unlink:
            samples_path.rmdir()


if __name__ == '__main__':
    tmax = 10
    n_jobs = -1

    run_samples(tmax, n_jobs=n_jobs)
    combine(unlink=True)
