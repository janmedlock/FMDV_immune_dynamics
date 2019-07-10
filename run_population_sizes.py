#!/usr/bin/python3

import os.path

import h5
import herd
import run_common


def run_population_size(model, SAT, population_size, tmax, nruns, hdfstore):
    p = herd.Parameters(model=model, SAT=SAT)
    p.population_size = population_size
    logging_prefix = (', '.join((f'model {model}',
                                 f'SAT {SAT}',
                                 f'population_size {population_size}'))
                      + ', ')
    df = run_common.run_many(p, tmax, nruns,
                             logging_prefix=logging_prefix)
    # Save the data for this `population_size`.
    # Add 'model', 'SAT', and 'population_size' levels to the index.
    run_common._prepend_index_levels(df, model=model, SAT=SAT,
                                     population_size=population_size)
    hdfstore.put(df, min_itemsize=run_common._min_itemsize)


def append_run_SATs(store_out, nruns):
    '''Append the data from 'run_SATs.h5' to `store_out`.'''
    p = herd.Parameters()
    where = f'run<{nruns}'
    with h5.HDFStore('run_SATs.h5', mode='r') as store_in:
        for chunk in store_in.select(where=where, iterator=True):
            # Add 'population_size' level to the index.
            run_common._insert_index_levels(chunk, 2,
                                            population_size=p.population_size)
            store_out.put(chunk, min_itemsize=run_common._min_itemsize)


if __name__ == '__main__':
    population_sizes = (
        # tuple(range(100, 1000, 100)) +
        # population_size = 1000 is in 'run_SATs.h5'.
        # `append_run_SATs()` will copy these into 'run_population_sizes.h5'.
        (2000, )
    )
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for population_size in population_sizes:
            for model in ('acute', 'chronic'):
                for SAT in (1, 2, 3):
                    run_population_size(model, SAT, population_size,
                                        tmax, nruns, store)
        # append_run_SATs(store, nruns)
        store.repack()
