#!/usr/bin/python3
'''Run simulations with varying susceptibility. This produces a file
called `susceptibility.h5`.'''

import common
import h5
import susceptibility


if __name__ == '__main__':
    NRUNS = 1000

    common.nice_self()
    with h5.HDFStore(susceptibility.store_path) as store:
        for SAT in common.SATs:
            for suscept in susceptibility.values:
                susceptibility.run(SAT, suscept, NRUNS, store)
        store.repack()
