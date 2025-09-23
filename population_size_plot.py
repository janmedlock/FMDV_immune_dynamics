#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size. This requires the file `population_size.h5`, which is
built by `population_size_run.py`.'''

import matplotlib.pyplot

import population_size
import sensitivity


if __name__ == '__main__':
    extinction_time = sensitivity.load(population_size)
    # sensitivity.plot_median(population_size, extinction_time)
    # sensitivity.plot_survival(population_size, extinction_time)
    # sensitivity.plot_kde(population_size, extinction_time)
    sensitivity.plot_kde_2d(population_size, extinction_time)
    matplotlib.pyplot.show()
