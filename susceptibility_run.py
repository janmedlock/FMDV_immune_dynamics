#!/usr/bin/python3
'''Run simulations with varying susceptibility. This produces a file
called `susceptibility.h5`.'''

import common
import sensitivity
import susceptibility


if __name__ == '__main__':
    NRUNS = 1000

    common.nice_self()
    sensitivity.run(susceptibility, NRUNS)
