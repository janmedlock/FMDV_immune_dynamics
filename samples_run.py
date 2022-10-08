#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples.h5`.'''

import samples


if __name__ == '__main__':
    N_JOBS = -1

    samples.run(n_jobs=N_JOBS)
    samples.combine(unlink=True)
