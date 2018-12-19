#!/usr/bin/python3

import os.path
import time

import numpy

import herd


def run_one(run_number, parameters, tmax, *args, **kwargs):
    '''Run one simulation.'''
    h = herd.Herd(parameters, run_number=run_number, *args, **kwargs)
    return h.run(tmax)


def make_plot(data, show = True):
    from matplotlib import pyplot
    import seaborn

    (fig, ax) = pyplot.subplots()
    seaborn.set_palette(seaborn.color_palette('deep', 6))

    for (k, x) in data.items():
        ax.step(365 * x.index, x, where = 'post', label = k)

    ax.set_xlabel(data.index.name)
    ax.set_ylabel('number')

    ax.legend()

    if show:
        pyplot.show()


if __name__ == '__main__':
    SAT = 1
    chronic = True
    seed = 1
    tmax = 10
    debug = False

    p = herd.Parameters(SAT=SAT, chronic=chronic)
    t0 = time.time()
    data = run_one(seed, p, tmax, debug=debug)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    make_plot(data)

    # _filebase, _ = os.path.splitext(__file__)
    # _picklefile = _filebase + '.pkl'
    # data.to_pickle(_picklefile)
