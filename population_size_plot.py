#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
population size. This requires the file `population_size.h5`, which is
built by `population_size_run.py`.'''

import matplotlib.pyplot

import population_size
import sensitivity


sens = sensitivity.Sensitivity(population_size,
                               'population_size',
                               'Population\nsize',
                               log=True)


if __name__ == '__main__':
    df = sens.load_extinction_time()
    # sens.plot_median(df)
    # sens.plot_survival(df)
    # sens.plot_kde(df)
    sens.plot_kde_2d(df)
    matplotlib.pyplot.show()
