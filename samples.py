'''Common code for running and plotting the parameter samples.'''

import itertools
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
    with joblib.Parallel(n_jobs=n_jobs,
                         return_as='generator') as parallel:
        results = parallel(
            joblib.delayed(run_one)(
                parameters, sample, sample_number, *args, **kwargs
            )
            for (sample_number, sample) in samples.iterrows()
        )
        chunker = itertools.batched(results, chunksize)
        for (chunk_number, chunk) in enumerate(chunker):
            # Make 'sample' the outer row index.
            start = chunk_number * chunksize
            end = min((chunk_number + 1) * chunksize, len(samples))
            sample_numbers = samples.index[start:end]
            yield pandas.concat(chunk, keys=sample_numbers, names=['sample'],
                                copy=False)


def run(SAT, samples, store, *args, **kwargs):
    '''Run simulation results with samples for SAT.'''
    parameters = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    chunks = run_many_chunked(parameters, samples, *args,
                              logging_prefix=logging_prefix, **kwargs)
    for chunk in chunks:
        common.prepend_index_levels(chunk, SAT=SAT)
        _save_result(store, chunk)
