#!/usr/bin/env python
'''Run simulations with varying probability of recovered true. This
produces a file called `recovered_true_probability.h5`.'''

import common
import recovered_true_probability
import sensitivity

if __name__ == '__main__':
    common.nice_self()
    sensitivity.run(recovered_true_probability, common.NRUNS)
