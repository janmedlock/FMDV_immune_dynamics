#!/usr/bin/python3

import numpy
from scipy import integrate
from matplotlib import pyplot
import seaborn

import sys
sys.path.append('..')

import herd
from herd import pde
from herd import birth


tmax = 10

agemax = 20
agestep = 0.1

parameters = herd.Parameters()
parameters.population_size = 10000

gap_sizes = (None, 0, 3, 6, 9) # In months.  None is aseasonal.


(fig, ax) = pyplot.subplots()
ax.set_xlabel('Time (years)')
ax.set_ylabel('Infected buffaloes')
colors = seaborn.color_palette('husl', n_colors = len(gap_sizes))

for (g, c) in zip(gap_sizes, colors):
    parameters.birth_seasonal_coefficient_of_variation \
        = birth.get_seasonal_coefficient_of_variation_from_gap_size(g)

    if g is None:
        label = 'Aseasonal'
    else:
        label = 'Seasonal, {}-month gap'.format(g)

    (t, ages, (M, S, I, R)) = pde.solve(tmax, agemax, agestep, parameters)

    i = integrate.trapz(I, ages, axis = 1)

    ax.plot(t, i, color = c, label = label)

ax.legend(loc = 'upper right')

pyplot.show()
