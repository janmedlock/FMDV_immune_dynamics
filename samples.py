'''Common code for running and plotting the parameter samples.'''

import pathlib

import joblib
import pandas

import baseline
import common
import herd
import herd.samples


store_path = pathlib.Path(__file__).with_suffix('.h5')


def load_samples():
    '''Load parameter samples.'''
    return herd.samples.load()


def _save_result(store, result):
    '''Save extinction time.'''
    common.save_result(store, result,
                       extinction_time=True)


def run_one(parameters, sample, sample_number, *args, **kwargs):
    '''Run one simulation.'''
    params = parameters.merge(**sample)
    return baseline.run_one(params, sample_number, *args, **kwargs)


def run_many_chunked(parameters, samples, *args,
                     chunksize=100, n_jobs=-1, **kwargs):
    '''Generator to return chunks of many simulation results.'''
    if chunksize < 1:
        chunksize = len(samples)
    starts = range(0, len(samples), chunksize)
    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        for start in starts:
            end = min(start + chunksize, len(samples))
            runs = range(start, end)
            results = parallel(
                joblib.delayed(run_one)(parameters, samples.loc[i], i,
                                        *args, **kwargs)
                for i in runs
            )
        # Make 'sample' the outer row index.
        yield pandas.concat(results, keys=runs, names=['sample'],
                            copy=False)


def run(SAT, samples, store, *args, **kwargs):
    parameters = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    chunks = run_many_chunked(parameters, samples, *args,
                              logging_prefix=logging_prefix, **kwargs)
    for chunk in chunks:
        common.prepend_index_levels(chunk, SAT=SAT)
        _save_result(store, chunk)
