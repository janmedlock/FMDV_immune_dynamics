#!/usr/bin/python3

import numpy
import time

import herd


def make_plot(data, show = True):
    from matplotlib import pyplot
    import seaborn
    from scipy import integrate

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
    tmax = 1
    seed = 1
    debug = False
    export_data = False

    p = herd.Parameters(SAT=SAT)

    t0 = time.time()
    data = herd.Herd(p, seed=seed, debug=debug).run(tmax)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    make_plot(data)
    if export_data:
        data.to_csv('run_one.csv')
