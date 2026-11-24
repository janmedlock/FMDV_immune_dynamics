#!/usr/bin/env python
'''Run simulations with varying susceptibility. This produces a file
called `lost_immunity_susceptibility.h5`.'''

import common
import lost_immunity_susceptibility
import sensitivity

if __name__ == '__main__':
    common.nice_self()
    sensitivity.run(lost_immunity_susceptibility, common.NRUNS)
