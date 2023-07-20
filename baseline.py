'''Common code for the running and plotting the baseline parameter
values.'''

import pathlib

from joblib import delayed, Parallel
import pandas

import common
import herd


store_path = pathlib.Path(__file__).with_suffix('.h5')


def seed_cache(parameters):
    '''Populate the cache.'''
    herd.RandomVariables(parameters)


def run_one(parameters, run_number, *args, tmax=None, **kwargs):
    '''Run one simulation.'''
    if tmax is None:
        tmax = common.TMAX
    herd_ = herd.Herd(parameters, run_number=run_number, *args, **kwargs)
    return herd_.run(tmax)


def run_many_chunked(parameters, nruns, *args,
                     chunksize=100, n_jobs=-1, **kwargs):
    '''Generator to return chunks of many simulation results.'''
    if chunksize < 1:
        chunksize = nruns
    seed_cache(parameters)
    starts = range(0, nruns, chunksize)
    for start in starts:
        end = min(start + chunksize, nruns)
        runs = range(start, end)
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(parameters, i, *args, **kwargs)
            for i in runs)
        # Make 'run' the outer row index.
        yield pandas.concat(results, keys=runs, names=['run'],
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


def run(SAT, nruns, hdfstore, _parameters=None, *args, **kwargs):
    if _parameters is None:
        _parameters = {}
    parameters = herd.Parameters(SAT=SAT, **_parameters)
    logging_prefix = f'{SAT=}'
    chunks = run_many_chunked(parameters, nruns, *args,
                              logging_prefix=logging_prefix, **kwargs)
    for dfr in chunks:
        common.prepend_index_levels(dfr, SAT=SAT)
        hdfstore.put(dfr)
