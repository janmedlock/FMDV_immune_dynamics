#!/usr/bin/python3
'''Run one simulation for each SAT.'''

import common
import h5
import simulation


if __name__ == '__main__':
    common.nice_self()
    with h5.HDFStore(simulation.store_path) as store:
        for SAT in common.SATs:
            simulation.run(SAT, store)
        store.repack()
        store.set_read_only()
