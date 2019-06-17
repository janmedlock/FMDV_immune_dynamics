#!/usr/bin/python3

import copy
import itertools
import os

from joblib import delayed, Parallel
import numpy
import pandas

import h5
import herd
import herd.samples
import run_common


_path = 'run_samples'
_index = 'time (y)'


def _run_sample(parameters, sample, tmax, path, sample_number, logging_prefix):
    '''Run one simulation.'''
    filename = os.path.join(path, f'{sample_number}.npy')
    if not os.path.exists(filename):
        p = copy.copy(parameters)
        for (k, v) in sample.items():
            setattr(p, k, v)
        try:
            # Catch AssertionError from failed optimization in
            # `herd._initial_conditions._find_hazard_infection()`.
            h = herd.Herd(p, run_number=sample_number,
                          logging_prefix=logging_prefix)
        except AssertionError:
            # Failed runs will get run with tighter integration tolerances
            # in `herd._initial_conditions`.
            pass
        else:
            df = h.run(tmax)
            # Save the data for this sample.
            numpy.save(filename, df.to_records())


def _run_samples_SAT(model, SAT, tmax, path, logging_prefix):
    '''Run many simulations in parallel.'''
    path_SAT = os.path.join(path, str(SAT))
    os.makedirs(path_SAT, exist_ok=True)
    logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
    parameters = herd.Parameters(model=model, SAT=SAT)
    samples = herd.samples.load(model=model, SAT=SAT)
    return (delayed(_run_sample)(parameters, s, tmax, path_SAT, i,
                                 logging_prefix_SAT)
            for (i, s) in samples.iterrows())


def _run_samples_model(model, tmax, path):
    path_model = os.path.join(path, model)
    os.makedirs(path_model, exist_ok=True)
    logging_prefix = f'model {model}, '
    return itertools.chain.from_iterable(
        _run_samples_SAT(model, SAT, tmax, path_model, logging_prefix)
        for SAT in run_common._SATs)


def run_samples(tmax):
    os.makedirs(_path, exist_ok=True)
    jobs = itertools.chain.from_iterable(
        _run_samples_model(model, tmax, _path)
        for model in run_common._models)
    Parallel(n_jobs=-1)(jobs)


def _get_sample_number(filename):
    base, _ = os.path.splitext(filename)
    return int(base)


def combine():
    with h5.HDFStore('run_samples.h5', mode='a') as store:
        # (model, SAT, sample) that are already in `store`.
        store_idx = store.get_index().droplevel(_index).unique()
        for model in os.listdir(_path):
            path_model = os.path.join(_path, model)
            for SAT in map(int, os.listdir(path_model)):
                path_SAT = os.path.join(path_model, str(SAT))
                # Sort in integer order.
                for filename in sorted(os.listdir(path_SAT),
                                       key=_get_sample_number):
                    sample = _get_sample_number(filename)
                    if (model, SAT, sample) not in store_idx:
                        path_sample = os.path.join(path_SAT, filename)
                        recarray = numpy.load(path_sample)
                        df = pandas.DataFrame.from_records(recarray,
                                                           index=_index)
                        run_common._prepend_index_levels(df,
                                                         model=model,
                                                         SAT=SAT,
                                                         sample=sample)
                        store.put(df, min_itemsize=run_common._min_itemsize)
                        # os.remove(path_sample)


if __name__ == '__main__':
    tmax = 10

    run_samples(tmax)
    # combine()
