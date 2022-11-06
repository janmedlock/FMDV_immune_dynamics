#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
susceptibility of the lost-immunity class. This requires the file
`susceptibility.h5`, which is built by `susceptibility_run.py`.'''

import matplotlib.pyplot

import sensitivity
import susceptibility


if __name__ == '__main__':
    df = sensitivity.load_extinction_time(susceptibility)
    # sensitivity.plot_median(susceptibility, df)
    # sensitivity.plot_survival(susceptibility, df)
    # sensitivity.plot_kde(susceptibility, df)
    sensitivity.plot_kde_2d(susceptibility, df)
    matplotlib.pyplot.show()
