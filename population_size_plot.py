#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size. This requires the file `population_size.h5`, which is
built by `population_size_run.py`.'''

import matplotlib.pyplot

import population_size
import sensitivity


if __name__ == '__main__':
    df = sensitivity.load_extinction_time(population_size)
    # sensitivity.plot_median(population_size, df)
    # sensitivity.plot_survival(population_size, df)
    # sensitivity.plot_kde(population_size, df)
    sensitivity.plot_kde_2d(population_size, df)
    matplotlib.pyplot.show()
