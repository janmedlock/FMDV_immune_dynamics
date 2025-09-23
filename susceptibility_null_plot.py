#!/usr/bin/python3
'''Analyze and plot the results of the simulations with 0
susceptibility of the lost-immunity class. This requires the file
`susceptibility_null.h5`, which is built by `susceptibility_null_run.py`.'''

import matplotlib.pyplot

import baseline_plot
import susceptibility_null


def load():
    return baseline_plot.load(_module=susceptibility_null)


def plot(infected, extinction_time,
         draft=False, save=True):
    return baseline_plot.plot(infected, extinction_time,
                              draft=draft, save=save,
                              _module=susceptibility_null)


if __name__ == '__main__':
    DRAFT = False
    (infected, extinction_time) = load()
    plot(infected, extinction_time, draft=DRAFT)
    matplotlib.pyplot.show()
