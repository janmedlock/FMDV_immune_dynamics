#!/usr/bin/python3

import numpy
from matplotlib import pyplot

import Parameters
from Parameters import birth_rectangular, birth_triangular, birth_sine

sigma_vals = numpy.linspace(0., 1., 6)

parameters = Parameters.Parameters()
RVs = Parameters.RandomVariables()

t = numpy.linspace(-0.5, 2.5, 301)

(fig, ax) = pyplot.subplots(3, 1)

lines = []
for sigma in sigma_vals:
    parameters.birthSeasonalVariance = sigma**2
    br = birth_rectangular.birth_gen(parameters,
                                     _findBirthScaling = False)
    bt = birth_triangular.birth_gen(parameters,
                                    _findBirthScaling = False)
    bs = birth_sine.birth_gen(parameters,
                              _findBirthScaling = False)

    l = ax[0].step(t, br.hazard(t, 0, 5), where = 'mid')
    l = ax[1].plot(t, bt.hazard(t, 0, 5))
    l = ax[2].plot(t, bs.hazard(t, 0, 5))
    lines.extend(l)

# Use most extreme y values from all the plots to set y axis limits.
yl = list(zip(*[a.get_ylim() for a in ax]))
yl = [numpy.min(yl[0]), numpy.max(yl[1])]
for a in ax:
    xl = a.get_xlim()
    xtickrange = list(map(int, (numpy.ceil(xl[0]), numpy.floor(xl[1]) + 1)))
    a.set_xticks(range(*xtickrange))
    a.set_xticklabels(['$t_0 {:+d}$'.format(i) if i != 0 else '$t_0$'
                       for i in range(*xtickrange)])

    ytickrange = list(map(int, (numpy.ceil(yl[0]), numpy.floor(yl[1]) + 1)))
    a.set_yticks(range(*ytickrange))
    a.set_yticklabels(['${:d} \mu$'.format(i) if i > 1 else
                       '$\mu$' if i == 1 else '$0$'
                       for i in range(*ytickrange)])

fig.legend(lines,
           ['$\sigma = {:g}$'.format(sigma) for sigma in sigma_vals],
           'lower center',
           ncol = len(sigma_vals))

pyplot.show()
