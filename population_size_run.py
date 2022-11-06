#!/usr/bin/python3
'''Run simulations with varying population size. This produces a file
called `population_size.h5`.'''

import common
import h5
import population_size


if __name__ == '__main__':
    NRUNS = 1000

    common.nice_self()
    with h5.HDFStore(population_size.store_path) as store:
        for SAT in common.SATs:
            for popsize in population_size.values:
                population_size.run(SAT, popsize, NRUNS, store)
        store.repack()
