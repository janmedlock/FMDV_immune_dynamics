#!/usr/bin/env python
'''Run one simulation.'''

import time

import matplotlib.pyplot
import numpy

from context import baseline, common, herd


def make_plot(data):
    (fig, axes) = matplotlib.pyplot.subplots()
    for (name, ser) in data.items():
        axes.plot(ser, label=name,
                  drawstyle='steps-post',
                  alpha=0.9, linewidth=1)
    axes.set_xlabel(data.index.name)
    axes.set_ylabel('number')
    axes.legend(loc='center right')
    return fig


if __name__ == '__main__':
    SAT = 1
    DEBUG = False

    rng = numpy.random.default_rng(common.SEED)
    p = herd.Parameters(SAT=SAT, seed=rng)
    t0 = time.time()
    data = baseline.run_one(p, seed=rng, debug=DEBUG)
    t = time.time() - t0
    print(f'Run time: {t} seconds.')

    make_plot(data)
    matplotlib.pyplot.show()
