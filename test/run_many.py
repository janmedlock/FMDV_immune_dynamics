#!/usr/bin/python3
'''Run many simulations.'''

import itertools
import time

from matplotlib import pyplot
import pandas
import seaborn

from context import baseline
from context import herd


def get_mean(data):
    t_mean = data.index.levels[1]
    data_mean = pandas.DataFrame(0, index=t_mean, columns=data.columns)
    persisting = pandas.Series(0, index=t_mean)
    for i in data.index.levels[0]:
        data_i = data.loc[i]
        # Only go to the end of this simulation.
        mask = (t_mean <= data_i.index[-1])
        data_mean.loc[mask] += data_i.reindex(t_mean[mask], method='ffill')
        persisting.loc[mask] += 1
    return data_mean.div(persisting, axis=0)


def make_plots(data, show=True):
    (fig, axes) = pyplot.subplots(6, sharex=True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))
    for (i, color) in zip(data.index.levels[0], colors):
        data_i = data.loc[i]
        for (j, (name, ser)) in enumerate(data_i.items()):
            axes[j].plot(data_i.index, ser,
                         drawstyle='steps-pre', color=color, alpha=0.5)
    data_mean = get_mean(data)
    for (j, (name, ser)) in enumerate(data_mean.items()):
        axes[j].plot(ser.index, ser,
                     drawstyle='steps-pre', color='black', alpha=0.6)
        axes[j].set_ylabel(name.replace(' ', '\n'))
    axes[-1].set_xlabel(data.index.names[1])
    for axes_ in axes:
        ylim = axes_.get_ylim()
        if ylim[0] < 0:
            axes_.set_ylim(ymin=0)
    if show:
        pyplot.show()
    return fig


if __name__ == '__main__':
    SAT = 1
    NRUNS = 100

    p = herd.Parameters(SAT=SAT)
    t0 = time.time()
    data = baseline.run_many(p, NRUNS)
    t = time.time() - t0
    print(f'Run time: {t} seconds.')

    make_plots(data)
