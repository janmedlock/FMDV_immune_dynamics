#!/usr/bin/env python
'''Analyze and plot the results of the simulations with varying
susceptibility of lost-immunity true and recovered false. This requires
the file `lost_immunity_susceptibility.h5`, which is built by
`lost_immunity_susceptibility_run.py`.'''

import lost_immunity_susceptibility
import sensitivity

if __name__ == '__main__':
    extinction_time = sensitivity.load(lost_immunity_susceptibility)
    sensitivity.plot_kde_2d(lost_immunity_susceptibility, extinction_time)
