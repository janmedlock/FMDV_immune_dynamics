#!/usr/bin/python3

from joblib import delayed, Parallel
import pandas

import h5
import herd


_SATs = (1, 2, 3)


def _insert_index_levels(df, i, **levels):
    df.index = pandas.MultiIndex.from_arrays(
        [df.index.get_level_values(n) for n in df.index.names[:i]]
        + [pandas.Index([v], name=k).repeat(len(df))
           for (k, v) in levels.items()]
        + [df.index.get_level_values(n) for n in df.index.names[i:]])


def _append_index_levels(df, **levels):
    _insert_index_levels(df, df.index.nlevels, **levels)


def _prepend_index_levels(df, **levels):
    _insert_index_levels(df, 0, **levels)


def run_one(parameters, tmax, run_number, *args, **kwargs):
    '''Run one simulation.'''
    h = herd.Herd(parameters, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def run_many(parameters, tmax, nruns, *args, **kwargs):
    '''Run many simulations in parallel.'''
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(parameters, tmax, i, *args, **kwargs)
        for i in range(nruns))
    # Make 'run' the outer row index.
    return pandas.concat(results, keys=range(nruns), names=['run'],
                         copy=False)


def run(SAT, tmax, nruns, hdfstore):
    p = herd.Parameters(SAT=SAT)
    logging_prefix = f'{SAT=}'
    df = run_many(p, tmax, nruns,
                  logging_prefix=logging_prefix)
    _prepend_index_levels(df, SAT=SAT)
    hdfstore.put(df)


if __name__ == '__main__':
    nruns = 1000
    tmax = 10

    filename = 'run.h5'
    with h5.HDFStore(filename) as store:
        for SAT in _SATs:
            run(SAT, tmax, nruns, store)
        store.repack()
