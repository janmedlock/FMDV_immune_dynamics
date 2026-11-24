#!/usr/bin/env python
'''Analyze and plot the results of the simulations with varying
probability of recovered true. This requires the file
`recovered_true_probability.h5`, which is built by
`recovered_true_probability_run.py`.'''

import recovered_true_probability
import sensitivity

if __name__ == '__main__':
    extinction_time = sensitivity.load(recovered_true_probability)
    sensitivity.plot_kde_2d(recovered_true_probability, extinction_time)
