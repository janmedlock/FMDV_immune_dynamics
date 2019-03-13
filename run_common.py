import copy
import time

from joblib import delayed, Parallel
import numpy
import pandas

import herd
import herd.samples


_SATs = (1, 2, 3)


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


def get_model(chronic):
    # Use a `pandas.CategoricalIndex()` to store the alternatives
    # to encode string widths for HDF.
    models = ['acute', 'chronic']
    val = model[1] if chronic else model[0]
    return pandas.CategoricalIndex([val], models, name='model')


def run_SATs(chronic, tmax, nruns, hdfstore, *args, **kwargs):
    for SAT in _SATs:
        p = herd.Parameters(chronic=chronic, SAT=SAT)
        print(f'Running SAT {SAT}.')
        t0 = time.time()
        df = run_many(p, tmax, nruns, *args, **kwargs)
        t1 = time.time()
        print(f'Run time: {t1 - t0} seconds.')
        # Save the data for this `SAT`.
        # Add 'model' and 'SAT' levels to the index.
        ix_model = get_model(chronic)
        # Leave enough space for all possible model names.
        min_itemsize = {ix_model.name: ix_model.categories.len().max()}
        ix_model = ix_model.astype('str')
        ix_SAT = pandas.Index([SAT], name='SAT')
        df.index = pandas.MultiIndex.from_arrays(
            [ix.repeat(len(df)) for ix in (ix_model, ix_SAT)]
            + [df.index.get_level_values(l) for l in df.index.names])
        hdfstore.put(df, format='table', append=True,
                     min_itemsize=min_itemsize)

def run_start_times(chronic, SAT, tmax, nruns, hdfstore, logging_prefix='',
                    *args, **kwargs):
    # Every month.
    for start_time in numpy.arange(0, 1, 1 / 12):
        p = herd.Parameters(chronic=chronic, SAT=SAT)
        p.start_time = start_time
        logging_prefix_ = (logging_prefix
                           + f'Start time {start_time * 12:g} / 12')
        print(f'Running {logging_prefix_}.')
        logging_prefix_ += ', '
        t0 = time.time()
        df = run_many(p, tmax, nruns,
                      logging_prefix=logging_prefix_,
                      *args, **kwargs)
        t1 = time.time()
        print(f'Run time: {t1 - t0} seconds.')
        # Save the data for this `start_time`.
        # Add 'model', 'SAT', and 'start_time' levels to the index.
        ix_model = get_model(chronic)
        # Leave enough space for all possible model names.
        min_itemsize = {ix_model.name: ix_model.categories.len().max()}
        ix_model = ix_model.astype('str')
        ix_SAT = pandas.Index([SAT], name='SAT')
        ix_start_time = pandas.Index([start_time], name='start_time')
        df.index = pandas.MultiIndex.from_arrays(
            [ix.repeat(len(df)) for ix in (ix_model, ix_SAT, ix_start_time)]
            + [df.index.get_level_values(l) for l in df.index.names])
        hdfstore.put(df, format='table', append=True,
                     min_itemsize=min_itemsize)


def run_start_times_SATs(chronic, tmax, nruns, hdfstore, *args, **kwargs):
    for SAT in _SATs:
        run_start_times(chronic, SAT, tmax, nruns, hdfstore,
                        logging_prefix=f'SAT {SAT}, ',
                        *args, **kwargs)


def _run_sample(parameters, sample, tmax, run_number, *args, **kwargs):
    '''Run one simulation.'''
    p = copy.copy(parameters)
    for k, v in sample.items():
        setattr(p, k, v)
    h = herd.Herd(p, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def _run_samples_SAT(chronic, SAT, tmax, *args, **kwargs):
    '''Run many simulations in parallel.'''
    samples = herd.samples.load(chronic=chronic, SAT=SAT)
    parameters = herd.Parameters(chronic=chronic, SAT=SAT)
    print(f'Running SAT {SAT}.')
    t0 = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(_run_sample)(parameters, s, tmax, i, *args, **kwargs)
        for i, s in samples.iterrows())
    t1 = time.time()
    print(f'Run time: {t1 - t0} seconds.')
    df = pandas.concat(results, keys=range(len(samples)),
                    names=['sample'], copy=False)
    # Add 'model' and 'SAT' levels to the index.
    ix_model = get_model(chronic)
    # Leave enough space for all possible model names.
    min_itemsize = {ix_model.name: ix_model.categories.len().max()}
    ix_model = ix_model.astype('str')
    ix_SAT = pandas.Index([SAT], name='SAT')
    df.index = pandas.MultiIndex.from_arrays(
        [ix.repeat(len(df)) for ix in (ix_model, ix_SAT)]
        + [df.index.get_level_values(l) for l in df.index.names])
    return df


def run_samples(chronic, tmax, hdfstore, *args, **kwargs):
    for SAT in (1, 2, 3):
        _run_samples_SAT(chronic, SAT, tmax, hdfstore, *args, **kwargs)
        # Save the data for this `SAT`.
        hdfstore.put(df, format='table', append=True,
                     min_itemsize=min_itemsize)
