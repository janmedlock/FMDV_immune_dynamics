import copy
import time

from joblib import delayed, Parallel
import numpy
import pandas

import herd
import herd.samples


_models = ('acute', 'chronic')
# Leave enough space in hdf for all model names.
_min_itemsize = {'model': max(len(m) for m in _models)}
_SATs = (1, 2, 3)


def _prepend_index_levels(df, **levels):
    df.index = pandas.MultiIndex.from_arrays(
        [pandas.Index([v], name=k).repeat(len(df)) for (k, v) in levels.items()]
        + [df.index.get_level_values(n) for n in df.index.names])


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


def run_SATs(model, tmax, nruns, hdfstore, logging_prefix='', *args, **kwargs):
    for SAT in _SATs:
        p = herd.Parameters(model=model, SAT=SAT)
        logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
        t0 = time.time()
        df = run_many(p, tmax, nruns,
                      logging_prefix=logging_prefix_SAT,
                      *args, **kwargs)
        t1 = time.time()
        print(f'Run time: {t1 - t0} seconds.')
        # Save the data for this `SAT`.
        # Add 'model' and 'SAT' levels to the index.
        _prepend_index_levels(df, model=model, SAT=SAT)
        hdfstore.put(df, format='table', append=True,
                     min_itemsize=_min_itemsize)


def run_start_times(model, SAT, tmax, nruns, hdfstore, logging_prefix='',
                    *args, **kwargs):
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(model=model, SAT=SAT)
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
        # Add 'model', 'SAT', and 'start_time' levels to the index.
        _prepend_index_levels(df, model=model, SAT=SAT, start_time=start_time)
        hdfstore.put(df, format='table', append=True,
                     min_itemsize=_min_itemsize)


def run_start_times_SATs(model, tmax, nruns, hdfstore, logging_prefix='',
                         *args, **kwargs):
    for SAT in _SATs:
        logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
        run_start_times(model, SAT, tmax, nruns, hdfstore,
                        logging_prefix=logging_prefix_SAT,
                        *args, **kwargs)


def _run_sample(parameters, sample, tmax, run_number, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for (k, v) in sample.items():
        setattr(p, k, v)
    h = herd.Herd(p, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def _run_samples_SAT(model, SAT, tmax, hdfstore, *args, save_every=100,
                     **kwargs):
    '''Run many simulations in parallel.'''
    samples = herd.samples.load(model=model, SAT=SAT)
    parameters = herd.Parameters(model=model, SAT=SAT)
    t0 = time.time()
    with Parallel(n_jobs=-1) as parallel:
        for start in range(0, len(samples), save_every):
            end = start + save_every
            idx = samples.index[start:end]
            results = parallel(
                delayed(_run_sample)(parameters, s, tmax, i, *args, **kwargs)
                for (i, s) in samples.loc[idx].iterrows())
            # Save the data for these samples.
            df = pandas.concat(results, keys=idx, copy=False)
            # Add 'model' and 'SAT' levels to the index.
            _prepend_index_levels(df, model=model, SAT=SAT)
            hdfstore.put(df, format='table', append=True,
                         min_itemsize=_min_itemsize)
    t1 = time.time()
    print(f'Run time: {t1 - t0} seconds.')


def run_samples(model, tmax, hdfstore, logging_prefix='', *args, **kwargs):
    for SAT in (1, 2, 3):
        logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
        _run_samples_SAT(model, SAT, tmax, hdfstore,
                         logging_prefix=logging_prefix_SAT,
                         *args, **kwargs)
