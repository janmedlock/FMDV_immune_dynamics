#!/usr/bin/python3

import os.path

import numpy

import h5
import herd
import run_common


def _copy_run_SATs(model, SAT, bscov, nruns, hdfstore_out):
    '''Copy the data from 'run_SATs.h5'.'''
    where = f'model={model} & SAT={SAT} & run<{nruns}'
    with h5.HDFStore('run_SATs.h5', mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run_common._insert_index_levels(
                chunk, 2,
                birth_seasonal_coefficient_of_variation=bscov)
            hdfstore_out.put(chunk, min_itemsize=run_common._min_itemsize)


def run_birth_seasonality(model, SAT, birth_seasonality_scaling, tmax, nruns,
                        hdfstore):
    p = herd.Parameters(model=model, SAT=SAT)
    p.birth_seasonal_coefficient_of_variation *= birth_seasonality_scaling
    bscov = p.birth_seasonal_coefficient_of_variation
    if birth_seasonality_scaling == 1:
        _copy_run_SATs(model, SAT, bscov, nruns, hdfstore)
    else:
        logging_prefix = (
            ', '.join((
                f'model {model}',
                f'SAT {SAT}',
                f'birth_seasonality_scaling {birth_seasonality_scaling}'))
            + ', ')
        df = run_common.run_many(p, tmax, nruns,
                                 logging_prefix=logging_prefix)
        run_common._prepend_index_levels(
            df, model=model, SAT=SAT,
            birth_seasonal_coefficient_of_variation=bscov)
        hdfstore.put(df, min_itemsize=run_common._min_itemsize)


if __name__ == '__main__':
    birth_scalings = numpy.linspace(0, 2, 5)
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for birth_scaling in birth_scalings:
            for model in ('acute', 'chronic'):
                for SAT in (1, 2, 3):
                    run_birth_seasonality(model, SAT, birth_scaling,
                                          tmax, nruns, store)
        store.repack()
