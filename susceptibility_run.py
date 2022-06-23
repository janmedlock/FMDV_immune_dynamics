#!/usr/bin/python3
'''Run simulations with varying susceptibility. This produces a file
called `susceptibility.h5`.'''

import common
import h5
import susceptibility


if __name__ == '__main__':
    tmax = 10
    nruns = 1000
    chunksize = 100
    n_jobs = -1

    with h5.HDFStore(store_path) as store:
        for SAT in common.SATs:
            for suscept in susceptibility.susceptibilities:
                susceptibility.run(SAT, suscept, tmax, nruns, store,
                                   chunksize=chunksize, n_jobs=n_jobs)
        store.repack()
