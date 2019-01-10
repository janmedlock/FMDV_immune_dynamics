#!/usr/bin/python3

import os.path
import time

from joblib import delayed, Parallel
import pandas

import h5
import herd
from run_one import run_one


def run_many(nruns, parameters, tmax, *args, **kwargs):
    '''Run many simulations in parallel.'''
    results = Parallel(n_jobs=-1)(
        delayed(run_one)(i, parameters, tmax, *args, **kwargs)
        for i in range(nruns))
    # Make 'run' the outer row index.
    return pandas.concat(results, keys=range(nruns), names=['run'],
                         copy=False)


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
    from matplotlib import pyplot
    import seaborn
    import itertools

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
    chronic = True
    nruns = 100
    tmax = 10

    p = herd.Parameters(SAT=SAT, chronic=chronic)
    t0 = time.time()
    data = run_many(nruns, p, tmax)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    make_plots(data)

    # _filebase, _ = os.path.splitext(__file__)
    # if chronic:
    #     _filebase += '_chronic'
    # _h5file = _filebase + '.h5'
    # h5.dump(data, _h5file)
