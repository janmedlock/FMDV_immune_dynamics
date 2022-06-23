#!/usr/bin/python3
import time

from matplotlib import pyplot

from context import baseline
from context import herd


def make_plot(data, show=True):
    (fig, ax) = pyplot.subplots()
    for (k, x) in data.items():
        ax.plot(x, label=k,
                drawstyle='steps-pre',
                alpha=0.9, linewidth=1)
    ax.set_xlabel(data.index.name)
    ax.set_ylabel('number')
    ax.legend(loc='center right')
    if show:
        pyplot.show()
    return fig


if __name__ == '__main__':
    SAT = 1
    tmax = 10
    seed = 1
    debug = False

    p = herd.Parameters(SAT=SAT)
    t0 = time.time()
    data = baseline.run_one(p, tmax, seed, debug=debug)
    t1 = time.time()
    print('Run time: {} seconds.'.format(t1 - t0))

    fig = make_plot(data)
