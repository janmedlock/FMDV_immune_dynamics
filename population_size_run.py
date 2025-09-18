#!/usr/bin/python3
'''Run simulations with varying population size. This produces a file
called `population_size.h5`.'''

import common
import population_size
import sensitivity


if __name__ == '__main__':
    common.nice_self()
    sensitivity.run(population_size, common.NRUNS)
