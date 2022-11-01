#!/usr/bin/python3
'''Analyze and plot the results of the simulations with varying
susceptibility of the lost-immunity class. This requires the file
`susceptibility.h5`, which is built by `susceptibility_run.py`.'''

import matplotlib.pyplot

import susceptibility
import sensitivity


sens = sensitivity.Sensitivity(susceptibility,
                               'lost_immunity_susceptibility',
                               'Susceptibility\nof lost-immunity\nstate',
                               log=False)


if __name__ == '__main__':
    df = sens.load_extinction_time()
    # sens.plot_median(df)
    # sens.plot_survival(df)
    # sens.plot_kde(df)
    sens.plot_kde_2d(df)
    matplotlib.pyplot.show()
