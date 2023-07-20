#!/usr/bin/python3
'''Run simulations with no chronic infections.'''


import acute
import common
import h5


if __name__ == '__main__':
    NRUNS = 1000

    common.nice_self()
    with h5.HDFStore(acute.store_path) as store:
        for SAT in common.SATs:
            acute.run(SAT, NRUNS, store)
        store.repack()
