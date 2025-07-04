#!/usr/bin/python3
'''Run simulations with the baseline parameter values. This produces a
file called `baseline.h5`.'''

import baseline
import common
import h5


if __name__ == '__main__':
    NRUNS = 1000

    common.nice_self()
    with h5.HDFStore(baseline.store_path) as store:
        for SAT in common.SATs:
            baseline.run(SAT, NRUNS, store)
        store.repack()
