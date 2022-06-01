#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples.h5`.'''


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


path_samples = pathlib.Path(__file__).parent / 'samples'
_t_name = 'time (y)'


def run_one(parameters, sample, tmax, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for (k, v) in sample.items():
        setattr(p, k, v)
    return run.run_one(p, tmax, sample_number, *args, **kwargs)


def run_one_and_save(parameters, sample, tmax, sample_number, path_dir, *args,
                     touch=True, **kwargs):
    '''Run one simulation.'''
    path_sample = path_dir / f'{sample_number}.npy'
    if not path_sample.exists():
        if touch:
            path_sample.touch(exist_ok=False)
        dfr = run_one(parameters, sample, tmax, path_sample,
                      *args, **kwargs)
        # Save the data for this sample.
        numpy.save(path_sample, dfr.to_records())


def _get_jobs_SAT(SAT, samples, tmax, path_dir, *args, **kwargs):
    '''Get jobs to run in parallel for one SAT.'''
    path_SAT = path_dir / str(SAT)
    path_SAT.makedir(exist_ok=True)
    p = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    return (delayed(run_one_and_save)(p, s, tmax, n, path_SAT, *args,
                                      logging_prefix=logging_prefix, **kwargs)
            for (n, s) in samples.iterrows())


def run_samples(tmax, *args,
                n_jobs=-1, **kwargs):
    samples = herd.samples.load()
    path_samples.makedir(exist_ok=True)
    jobs = itertools.chain.from_iterable(
        _get_jobs_SAT(SAT, samples[SAT], tmax, path_samples, *args, **kwargs)
        for SAT in (3, 2, 1))
    Parallel(n_jobs=n_jobs)(jobs)


def _get_SAT(path):
    return int(path.name)


def _get_sample_number(path):
    return int(path.stem)


def combine(unlink=True):
    path_store = pathlib.Path('samples.h5')
    with h5.HDFStore(path_store, mode='a') as store:
        # (SAT, sample) that are already in `store`.
        store_idx = store.get_index().droplevel(_t_name).unique()
        paths_SAT = sorted(path_samples.iterdir(), key=_get_SAT)
        for path_SAT in paths_SAT:
            SAT = _get_SAT(path_SAT)
            paths_sample = sorted(path_SAT.iterdir(), key=_get_sample_number)
            for path_sample in paths_sample:
                sample = _get_sample_number(path_sample)
                if (SAT, sample) not in store_idx:
                    recarray = numpy.load(path_sample)
                    dfr = pandas.DataFrame.from_records(recarray,
                                                        index=_t_name)
                    run.prepend_index_levels(dfr, SAT=SAT, sample=sample)
                    print('Inserting '
                          + ', '.join((f'SAT={SAT}',
                                       f'sample={sample}'))
                          + '.')
                    store.put(dfr)
                if unlink:
                    path_sample.unlink()
            if unlink:
                path_SAT.rmdir()
        if unlink:
            path_samples.rmdir()


if __name__ == '__main__':
    tmax = 10

    run_samples(tmax)
    # combine(unlink=False)
