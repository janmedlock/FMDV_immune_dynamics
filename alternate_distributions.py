#!/usr/bin/python

import numpy
from matplotlib import pyplot

def get_rectangular_hazard(mu, sigma):
    a = (mu ** 2 + sigma ** 2) / mu
    b = mu ** 2 / (mu ** 2 + sigma ** 2)

    def f(t):
        tau = numpy.mod(t + 0.5, 1) - 0.5
        return numpy.where(numpy.abs(tau) <= b / 2.,
                           a, 0.)

    return f


def get_triangular_hazard(mu, sigma):
    if sigma < mu / numpy.sqrt(3):
        a = mu + sigma * numpy.sqrt(3)
        b = 2 * sigma * numpy.sqrt(3) / (mu + sigma * numpy.sqrt(3))
    else:
        a = 3 * (mu ** 2 + sigma ** 2) / 2 / mu
        b = 3 * (mu ** 2 + sigma ** 2) / 4 / mu ** 2

    def f(t):
        tau = numpy.mod(t + 0.5, 1) - 0.5
        return numpy.clip(a * (1 - 2 * b * numpy.abs(tau)),
                          0, numpy.inf)

    return f


mu = 1.
sigma_vals = numpy.linspace(0., 1., 6)

t = numpy.linspace(-0.5, 2.5, 301)

(fig, ax) = pyplot.subplots(2, 1)

lines = []
for sigma in sigma_vals:
    f = get_rectangular_hazard(mu, sigma)
    g = get_triangular_hazard(mu, sigma)
    l = ax[0].step(t, f(t), where = 'mid')
    l = ax[1].plot(t, g(t))
    lines.extend(l)

for a in ax:
    xl = a.get_xlim()
    xtickrange = map(int, (numpy.ceil(xl[0]), numpy.floor(xl[1]) + 1))
    a.set_xticks(range(*xtickrange))
    a.set_xticklabels(['$t_0 {:+d}$'.format(i) if i != 0 else '$t_0$'
                       for i in range(*xtickrange)])

    yl = a.get_ylim()
    ytickrange = map(int, (numpy.ceil(yl[0]), numpy.floor(yl[1]) + 1))
    a.set_yticks(range(*ytickrange))
    a.set_yticklabels(['${:d} \mu$'.format(i) if i > 1 else
                       '$\mu$' if i == 1 else '$0$'
                       for i in range(*ytickrange)])

fig.legend(lines,
           ['$\sigma = {:g}$'.format(sigma) for sigma in sigma_vals],
           'lower center',
           ncol = len(sigma_vals))

pyplot.show()
