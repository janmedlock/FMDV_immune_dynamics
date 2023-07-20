#!/usr/bin/python3
'''Build a figure comparing the runs without chronic infections. This
requires the file `acute.h5`, which is built by `acute_run.py`.'''

from matplotlib import pyplot

import acute
import baseline_plot


def load():
    return baseline_plot.load(_module=acute)


def plot(infected, extinction_time,
         draft=False, save=True):
    return baseline_plot.plot(infected, extinction_time,
                              draft=draft, save=save,
                              _module=acute)


if __name__ == '__main__':
    DRAFT = False
    (infected, extinction_time) = load()
    plot(infected, extinction_time, draft=DRAFT)
    pyplot.show()
