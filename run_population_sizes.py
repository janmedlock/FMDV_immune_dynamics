#!/usr/bin/python3

import os.path

import numpy

import h5
import herd
import run_common


def _copy_run_SATs(SAT, population_size, nruns, hdfstore_out):
    '''Copy the data from 'run_SATs.h5'.'''
    where = f'SAT={SAT} & run<{nruns}'
    with h5.HDFStore('run_SATs.h5', mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run_common._insert_index_levels(chunk, 2,
                                            population_size=population_size)
            hdfstore_out.put(chunk, min_itemsize=run_common._min_itemsize)


def run_population_size(SAT, population_size, tmax, nruns, hdfstore):
    if population_size == 1000:
        _copy_run_SATs(SAT, population_size, nruns, hdfstore)
    else:
        p = herd.Parameters(SAT=SAT)
        p.population_size = population_size
        logging_prefix = (', '.join((f'SAT {SAT}',
                                     f'population_size {population_size}'))
                          + ', ')
        df = run_common.run_many(p, tmax, nruns,
                                 logging_prefix=logging_prefix)
        run_common._prepend_index_levels(df, SAT=SAT,
                                         population_size=population_size)
        hdfstore.put(df, min_itemsize=run_common._min_itemsize)


def arange(start, stop, step):
    '''Like `numpy.arange()`, but with `stop` in the range.'''
    return numpy.arange(start, stop + step, step)


if __name__ == '__main__':
    population_sizes = numpy.hstack((arange(100, 900, 100),
                                     arange(1000, 5000, 1000)))
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for population_size in population_sizes:
            for SAT in (1, 2, 3):
                run_population_size(SAT, population_size,
                                    tmax, nruns, store)
        store.repack()
