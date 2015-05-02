#!/usr/bin/python

import numpy
from scipy import integrate
from matplotlib import pyplot
import seaborn

import Parameters
from Parameters import pde


tMax = 10.

ageMax = 20.
ageStep = 0.1

parameters = Parameters.Parameters()
parameters.populationSize = 10000
parameters.infectionDuration = 21. / 365.
parameters.R0 = 10.

gapSizes = (None, 0, 3, 6, 9) # In months.  None is aseasonal.


(fig, ax) = pyplot.subplots()
ax.set_xlabel('Time (years)')
ax.set_ylabel('Infected buffaloes')
colors = seaborn.color_palette('husl', n_colors = len(gapSizes))

for (g, c) in zip(gapSizes, colors):
    if g is None:
        parameters.birthSeasonalVariance = 0.
        label = 'Aseasonal'
    else:
        parameters.birthSeasonalVariance = 4. / 3. / (1. - g / 12.) - 1.
        label = 'Seasonal, {}-month gap'.format(g)

    (t, ages, (M, S, I, R)) = pde.solve(tMax, ageMax, ageStep, parameters)

    i = integrate.trapz(I, ages, axis = 1)

    ax.plot(t, i, color = c, label = label)

ax.legend(loc = 'upper right')

pyplot.show()
