#!/usr/bin/python3

import itertools
import time

from joblib import delayed, Parallel
from matplotlib import pyplot
import pandas
import seaborn

import herd
import run_common


def get_mean(data):
    t_mean = data.index.levels[1]
    data_mean = pandas.DataFrame(0, index=t_mean, columns=data.columns)
    n = pandas.Series(0, index=t_mean)
    for i in data.index.levels[0]:
        data_i = data.loc[i]
        # Only go to the end of this simulation.
        mask = (t_mean <= data_i.index[-1])
        data_mean.loc[mask] += data_i.reindex(t_mean[mask], method='ffill')
        n.loc[mask] += 1
    return data_mean.div(n, axis=0)


def make_plots(data, show=True):
    (fig, ax) = pyplot.subplots(6, sharex=True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))
    for (i, color) in zip(data.index.levels[0], colors):
        data_i = data.loc[i]
        for (j, (k, x)) in enumerate(data_i.items()):
            ax[j].step(365 * data_i.index, x,
                       where='post', color=color, alpha=0.5)

    data_mean = get_mean(data)
    for (j, (k, x)) in enumerate(data_mean.items()):
        ax[j].step(365 * x.index, x,
                   where='post', color='black', alpha=0.6)
        ax[j].set_ylabel(k.replace(' ', '\n'))

    ax[-1].set_xlabel(data.index.names[1])

    for ax_ in ax:
        yl = ax_.get_ylim()
        if yl[0] < 0:
            ax_.set_ylim(ymin=0)

    if show:
        pyplot.show()


if __name__ == '__main__':
    SAT = 1
    model = 'chronic'
    nruns = 100
    tmax = 10

    p = herd.Parameters(model=model, SAT=SAT)
    t0 = time.time()
    data = run_common.run_many(p, tmax, nruns)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    make_plots(data)
