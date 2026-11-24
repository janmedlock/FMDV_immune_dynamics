#!/usr/bin/env python
'''Run simulations with varying probability of lost-immunity
true. This produces a file called `lost_immunity_true_probability.h5`.'''

import common
import lost_immunity_true_probability
import sensitivity

if __name__ == '__main__':
    common.nice_self()
    sensitivity.run(lost_immunity_true_probability, common.NRUNS)
