import time

from joblib import delayed, Parallel
import numpy
import pandas

import herd


_SATs = (1, 2, 3)


def _insert_index_levels(df, i, **levels):
    df.index = pandas.MultiIndex.from_arrays(
        [df.index.get_level_values(n) for n in df.index.names[:i]]
        + [pandas.Index([v], name=k).repeat(len(df))
           for (k, v) in levels.items()]
        + [df.index.get_level_values(n) for n in df.index.names[i:]])


def _append_index_levels(df, **levels):
    _insert_index_levels(df, 0, **levels)


def _prepend_index_levels(df, **levels):
    _insert_index_levels(df, df.index.nlevels, **levels)


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


def run_SATs(tmax, nruns, hdfstore, logging_prefix='', *args, **kwargs):
    for SAT in _SATs:
        p = herd.Parameters(SAT=SAT)
        logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
        t0 = time.time()
        df = run_many(p, tmax, nruns,
                      logging_prefix=logging_prefix_SAT,
                      *args, **kwargs)
        t1 = time.time()
        print(f'Run time: {t1 - t0} seconds.')
        # Save the data for this `SAT`.
        # Add 'SAT' levels to the index.
        _prepend_index_levels(df, SAT=SAT)
        hdfstore.put(df, min_itemsize=_min_itemsize)


def run_start_times(SAT, tmax, nruns, hdfstore, logging_prefix='',
                    *args, **kwargs):
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(SAT=SAT)
        p.start_time = start_time
        logging_prefix_start = (logging_prefix
                                + f'Start time {start_time * 12:g} / 12, ')
        t0 = time.time()
        df = run_many(p, tmax, nruns,
                      logging_prefix=logging_prefix_start,
                      *args, **kwargs)
        t1 = time.time()
        print(f'Run time: {t1 - t0} seconds.')
        # Save the data for this `start_time`.
        # Add 'SAT' and 'start_time' levels to the index.
        _prepend_index_levels(df, SAT=SAT, start_time=start_time)
        hdfstore.put(df, min_itemsize=_min_itemsize)


def run_start_times_SATs(tmax, nruns, hdfstore, logging_prefix='',
                         *args, **kwargs):
    for SAT in _SATs:
        logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
        run_start_times(SAT, tmax, nruns, hdfstore,
                        logging_prefix=logging_prefix_SAT,
                        *args, **kwargs)
