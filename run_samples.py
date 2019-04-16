#!/usr/bin/python3

import copy
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
    p = copy.copy(parameters)
    for (k, v) in sample.items():
        setattr(p, k, v)
    h = herd.Herd(p, run_number=sample_number, logging_prefix=logging_prefix)
    df = h.run(tmax)
    df.reset_index(inplace=True)
    # Save the data for this sample.
    filename = os.path.join(path, f'{sample_number}.npy')
    numpy.save(filename, df.to_records())


def _run_samples_SAT(model, SAT, tmax, path, logging_prefix):
    '''Run many simulations in parallel.'''
    path_SAT = os.path.join(path, str(SAT))
    os.makedirs(path_SAT, exist_ok=True)
    logging_prefix_SAT = logging_prefix + f'SAT {SAT}, '
    samples = herd.samples.load(model=model, SAT=SAT)
    parameters = herd.Parameters(model=model, SAT=SAT)
    Parallel(n_jobs=-1)(
        delayed(_run_sample)(parameters, s, tmax, path, i, logging_prefix_SAT)
        for (i, s) in samples.iterrows())


def _run_samples_model(model, tmax, path):
    path_model = os.path.join(path, model)
    os.makedirs(path_model, exist_ok=True)
    logging_prefix = f'model {model}, '
    for SAT in run_common._SATs:
        _run_samples_SAT(model, SAT, tmax, path_model, logging_prefix)


def run_samples(tmax):
    os.makedirs(_path, exist_ok=True)
    for model in run_common._models:
        _run_samples_model(model, tmax, _path)


def _get_sample_number(filename):
    base, _ = os.path.splitext(filename)
    return int(base)


def combine():
    with h5.HDFStore('run_samples.h5', mode='r') as store:
        idx = store.get_index().droplevel(_index).unique()
        for model in os.listdir(_path):
            path_model = os.path.join(path, model)
            for SAT_s in os.listdir(path_model):
                SAT = int(SAT_s)
                path_SAT = os.path.join(path_model, SAT_s)
                # Sort in integer order.
                for filename in sorted(os.listdir(path_SAT),
                                       key=_get_sample_number):
                    sample = _get_sample_number(filename)
                    if (model, SAT, sample) not in idx:
                        path_sample = os.path.join(path_SAT, filename)
                        df = pandas.DataFrame(numpy.load(path_sample))
                        df.set_index(_index, inplace=True)
                        run_common._prepend_index_levels(df,
                                                         model=model,
                                                         SAT=SAT,
                                                         sample=sample)
                        store.put(df, format='table', append=True,
                                  min_itemsize=run_common._min_itemsize)


if __name__ == '__main__':
    tmax = 10

    run_samples(tmax)
