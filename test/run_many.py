#!/usr/bin/env python
'''Run many simulations.'''

import itertools
import os
import pathlib
import sys
import time

import matplotlib.pyplot
import numpy
import pandas
import seaborn

from _context import baseline, common, herd


def run_many(parameters, nruns, seed=None):
    # Add the parent directory to `sys.path` so that the
    # `joblib.Parallel()` workers can find the `herd` module.
    path = pathlib.Path(__file__).parents[1]
    sys.path.append(str(path))
    data = baseline.run_many(parameters, nruns, seed=rng)
    sys.path.pop()
    return data


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


def make_plots(data):
    (fig, axes) = matplotlib.pyplot.subplots(data.shape[1], sharex=True)
    colors = itertools.cycle(seaborn.color_palette('husl', 8))
    plot_kwds = {
        'drawstyle': 'steps-post',
    }
    for (i, color) in zip(data.index.levels[0], colors):
        data_i = data.loc[i]
        for (j, (name, ser)) in enumerate(data_i.items()):
            axes[j].plot(data_i.index, ser,
                         color=color, alpha=0.5,
                         **plot_kwds)
    data_mean = get_mean(data)
    for (j, (name, ser)) in enumerate(data_mean.items()):
        axes[j].plot(ser.index, ser,
                     color='black', alpha=0.6,
                     **plot_kwds)
        axes[j].set_ylabel(name.replace(' ', '\n'))
    axes[-1].set_xlabel(data.index.names[1])
    for axes_ in axes:
        ylim = axes_.get_ylim()
        if ylim[0] < 0:
            axes_.set_ylim(ymin=0)
    fig.align_ylabels()
    return fig


if __name__ == '__main__':
    SAT = 1
    nruns = max(os.process_cpu_count(), 2)

    rng = numpy.random.default_rng(common.SEED)
    parameters = herd.Parameters(SAT=SAT, seed=rng)
    t0 = time.time()
    data = run_many(parameters, nruns, seed=rng)
    t = time.time() - t0
    print(f'Run time: {t} seconds.')

    make_plots(data)
    matplotlib.pyplot.show()
