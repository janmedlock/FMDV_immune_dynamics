'''Common code for the running and plotting the baseline parameter
values.'''

import itertools
import pathlib

import joblib
import pandas

import common
import herd
import logging_


store_path = pathlib.Path(__file__).with_suffix('.h5')


def _save_result(store, result):
    '''Save infected daily and extinction time.'''
    common.save_result(store, result,
                       infected_daily=True, extinction_time=True)


def seed_cache(parameters):
    '''Populate the cache.'''
    herd.RandomVariables(parameters)


def run_one(parameters, *args, tmax=None,
            print_duration=False, logging_prefix=None,
            run_number=None, **kwargs):
    '''Run one simulation.'''
    if tmax is None:
        tmax = common.TIME_MAX
    if run_number is not None:
        logging_prefix = logging_.append_to_prefix(logging_prefix,
                                                   run_number=run_number)
    herd_ = herd.Herd(parameters, *args, **kwargs)
    return herd_.run(tmax,
                     print_duration=print_duration,
                     logging_prefix=logging_prefix)


def run_many_chunked(parameters, nruns, *args,
                     chunksize=100, n_jobs=-1,
                     debug=False, print_duration=None, **kwargs):
    '''Generator to return chunks of many simulation results.'''
    if chunksize < 1:
        chunksize = nruns
    if print_duration is None:
        print_duration = not debug
    seed_cache(parameters)
    with joblib.Parallel(n_jobs=n_jobs,
                         return_as='generator') as parallel:
        results = parallel(
            joblib.delayed(run_one)(
                parameters, *args,
                debug=debug, print_duration=print_duration,
                run_number=run_number, **kwargs
            )
            for run_number in range(nruns)
        )
        chunker = itertools.batched(results, chunksize)
        for (chunk_number, chunk) in enumerate(chunker):
            # Make 'run' the outer row index.
            start = chunk_number * chunksize
            end = min((chunk_number + 1) * chunksize, nruns)
            run_numbers = range(start, end)
            yield pandas.concat(chunk, keys=run_numbers, names=['run'],
                                copy=False)


def run_many(parameters, nruns, *args, **kwargs):
    '''Run many simulations in parallel.'''
    chunks = run_many_chunked(parameters, nruns, *args, **kwargs)
    # Everything was run in one chunk.
    results = next(chunks)
    try:
        next(chunks)
    except StopIteration:
        pass
    else:
        raise RuntimeError('`chunks` has length > 1!')
    return results


def run(SAT, nruns, store, _parameters=None, *args, **kwargs):
    '''Run `nruns` for `SAT`.'''
    if _parameters is None:
        _parameters = {}
    parameters = herd.Parameters(SAT=SAT, **_parameters)
    logging_prefix = f'{SAT=}'
    chunks = run_many_chunked(parameters, nruns, *args,
                              logging_prefix=logging_prefix, **kwargs)
    for chunk in chunks:
        common.prepend_index_levels(chunk, SAT=SAT)
        _save_result(store, chunk)
