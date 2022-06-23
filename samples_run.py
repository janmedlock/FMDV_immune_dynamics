#!/usr/bin/python3
'''For each of the 3 SATs and for each of 20,000 posterior parameter
estimates, run 1 simulation. This produces a file called
`samples.h5`.'''

import samples


if __name__ == '__main__':
    tmax = 10
    n_jobs = 20

    samples.run(tmax, n_jobs=n_jobs)
    # samples.combine(unlink=False)
