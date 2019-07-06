#!/usr/bin/python3

import os.path

import h5
import herd
import run_common


def run_population_size(model, SAT, population_size, tmax, nruns, hdfstore):
    p = herd.Parameters(model=model, SAT=SAT)
    p.population_size = population_size
    logging_prefix = ', '.join((f'model {model}',
                                f'SAT {SAT}',
                                f'population_size {population_size}'))
    df = run_common.run_many(p, tmax, nruns,
                             logging_prefix=logging_prefix)
    # Save the data for this `population_size`.
    # Add 'model', 'SAT', and 'population_size' levels to the index.
    run_common._prepend_index_levels(df, model=model, SAT=SAT,
                                     population_size=population_size)
    hdfstore.put(df, min_itemsize=run_common._min_itemsize)


if __name__ == '__main__':
    population_sizes = (
        tuple(range(100, 1000, 100))
        # + tuple(range(1000, 10000, 1000))
    )
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for model in ('acute', 'chronic'):
            for SAT in (1, 2, 3):
                for population_size in population_sizes:
                    run_population_size(model, SAT, population_size,
                                        tmax, nruns, store)
        store.repack(_filename)
