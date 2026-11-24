#!/usr/bin/env python
'''Analyze and plot the results of the simulations with varying
probability of lost-immunity true. This requires the file
`lost_immunity_true_probability.h5`, which is built by
`lost_immunity_true_probability_run.py`.'''

import lost_immunity_true_probability
import sensitivity

if __name__ == '__main__':
    extinction_time = sensitivity.load(lost_immunity_true_probability)
    sensitivity.plot_kde_2d(lost_immunity_true_probability, extinction_time)
