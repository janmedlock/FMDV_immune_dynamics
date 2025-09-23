#!/usr/bin/python3
'''Run simulations with 0 susceptibility of the lost-immunity
class. This produces a file called `susceptibility_null.h5`.'''

import susceptibility_null
import common
import h5


if __name__ == '__main__':
    common.nice_self()
    with h5.HDFStore(susceptibility_null.store_path) as store:
        for SAT in common.SATs:
            susceptibility_null.run(SAT, common.NRUNS, store)
        store.repack()
        store.set_read_only()
