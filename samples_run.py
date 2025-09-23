#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples.h5`.'''

import common
import h5
import samples


if __name__ == '__main__':
    N_JOBS = -1

    common.nice_self()
    parameter_samples = samples.load_samples()
    with h5.HDFStore(samples.store_path) as store:
        for SAT in common.SATs:
            samples.run(SAT, parameter_samples[SAT], store, n_jobs=N_JOBS)
        store.repack()
        store.set_read_only()
