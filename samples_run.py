#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples_run.h5`.'''


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


def _SAT_path(SAT):
    return samples_path.joinpath(str(SAT))


def _sample_path(SAT, sample_number):
    return _SAT_path(SAT).joinpath(f'{sample_number}.npy')


def run_one(parameters, sample, tmax, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    params = parameters.merge(**sample)
    return run.run_one(params, tmax, sample_number, *args, **kwargs)


def run_one_and_save(parameters, sample, tmax, sample_number, path, *args,
                     touch=True, **kwargs):
    '''Run one simulation and save the output.'''
    if not path.exists():
        if touch:
            path.touch(exist_ok=False)
        dfr = run_one(parameters, sample, tmax, sample_number,
                      *args, **kwargs)
        # Save the data for this sample.
        numpy.save(path, dfr.to_records())


def run_samples(tmax, *args,
                n_jobs=-1, **kwargs):
    samples = herd.samples.load()
    parameters = {SAT: herd.Parameters(SAT=SAT)
                  for SAT in run.SATs}
    samples_path.mkdir(exist_ok=True)
    for SAT in run.SATs:
        _SAT_path(SAT).mkdir(exist_ok=True)
    # For each sample, iterate over the SATs,
    # i.e. SAT changes fastest.
    jobs = (delayed(run_one_and_save)(parameters[SAT], sample[SAT], tmax,
                                      sample_number,
                                      _sample_path(SAT, sample_number),
                                      *args, logging_prefix=f'{SAT=}',
                                      **kwargs)
            for (sample_number, sample) in samples.iterrows()
            for SAT in run.SATs)
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
    n_jobs = 20

    run_samples(tmax, n_jobs=n_jobs)
    # combine(unlink=False)
