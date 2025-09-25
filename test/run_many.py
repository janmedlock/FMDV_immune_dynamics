#!/usr/bin/python3
'''Run many simulations.'''

import itertools
import pathlib
import sys
import time

import matplotlib.pyplot
import pandas
import seaborn

from context import baseline
from context import herd


def run_many(parameters, nruns):
    # Add the parent directory to `sys.path` so that the
    # `joblib.Parallel()` workers can find the `herd` module.
    path = pathlib.Path(__file__).parents[1]
    sys.path.append(str(path))
    data = baseline.run_many(parameters, nruns)
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
    return fig


if __name__ == '__main__':
    SAT = 1
    NRUNS = 100

    parameters = herd.Parameters(SAT=SAT)
    t0 = time.time()
    data = run_many(parameters, NRUNS)
    t = time.time() - t0
    print(f'Run time: {t} seconds.')

    make_plots(data)
    matplotlib.pyplot.show()
