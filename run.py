#!/usr/bin/python3

import itertools
import pathlib

from joblib import delayed, Parallel
import pandas

import h5
import herd


SATs = (1, 2, 3)


store_path = pathlib.Path(__file__).with_suffix('.h5')


def insert_index_levels(dfr, i, **levels):
    dfr.index = pandas.MultiIndex.from_arrays(
        [dfr.index.get_level_values(n) for n in dfr.index.names[:i]]
        + [pandas.Index([v], name=k).repeat(len(dfr))
           for (k, v) in levels.items()]
        + [dfr.index.get_level_values(n) for n in dfr.index.names[i:]])


def append_index_levels(dfr, **levels):
    insert_index_levels(dfr, dfr.index.nlevels, **levels)


def prepend_index_levels(dfr, **levels):
    insert_index_levels(dfr, 0, **levels)


def seed_cache(SAT, **kwds):
    '''Populate the cache.'''
    parameters = herd.Parameters(SAT=SAT, **kwds)
    herd.RandomVariables(parameters)


def run_one(parameters, tmax, run_number, *args, **kwargs):
    '''Run one simulation.'''
    h = herd.Herd(parameters, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def run_many_chunked(parameters, tmax, nruns, *args,
                     chunksize=-1, n_jobs=-1, **kwargs):
    '''Generator to return chunks of many simulation results.'''
    if chunksize < 1:
        chunksize = nruns
    seed_cache(parameters)
    starts = range(0, nruns, chunksize)
    for start in starts:
        end = min(start + chunksize, nruns)
        runs = range(start, end)
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(parameters, tmax, i, *args, **kwargs)
            for i in runs)
        # Make 'run' the outer row index.
        yield pandas.concat(results, keys=runs, names=['run'],
                            copy=False)


def run_many(parameters, tmax, nruns, *args,
             n_jobs=-1, **kwargs):
    '''Run many simulations in parallel.'''
    chunks = run_many_chunked(parameters, tmax, nruns, *args,
                              n_jobs=n_jobs, **kwargs)
    # Everything was run in one chunk.
    results = next(chunks)
    try:
        next(chunks)
    except StopIteration:
        pass
    else:
        raise RuntimeError('`chunks` has length > 1!')
    return results


def run(SAT, tmax, nruns, hdfstore, *args,
        chunksize=-1, n_jobs=-1, **kwargs):
    parameters = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    chunks = run_many_chunked(parameters, tmax, nruns, *args,
                              chunksize=chunksize, n_jobs=n_jobs,
                              logging_prefix=logging_prefix, **kwargs)
    for dfr in chunks:
        prepend_index_levels(dfr, SAT=SAT)
        hdfstore.put(dfr)


if __name__ == '__main__':
    tmax = 10
    nruns = 1000
    chunksize = 100
    n_jobs = -1

    with h5.HDFStore(store_path) as store:
        for SAT in SATs:
            seed_cache(SAT)
            run(SAT, tmax, nruns, store,
                chunksize=chunksize, n_jobs=n_jobs)
        store.repack()
