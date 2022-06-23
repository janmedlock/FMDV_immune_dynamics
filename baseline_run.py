#!/usr/bin/python3
'''Run simulations with the baseline parameter values.'''


import baseline
import common
import h5


if __name__ == '__main__':
    tmax = 10
    nruns = 1000
    chunksize = 100
    n_jobs = -1

    with h5.HDFStore(baseline.store_path) as store:
        for SAT in common.SATs:
            baseline.run(SAT, tmax, nruns, store,
                         chunksize=chunksize, n_jobs=n_jobs)
        store.repack()
