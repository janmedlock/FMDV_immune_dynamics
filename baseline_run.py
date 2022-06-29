#!/usr/bin/python3
'''Run simulations with the baseline parameter values.'''


import baseline
import common
import h5


if __name__ == '__main__':
    NRUNS = 1000

    with h5.HDFStore(baseline.store_path) as store:
        for SAT in common.SATs:
            baseline.run(SAT, NRUNS, store)
        store.repack()
