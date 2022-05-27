#!/usr/bin/python3
from joblib import delayed, Parallel
import numpy
from matplotlib import pyplot

from context import herd
import herd.age_structure


start_times = numpy.linspace(0, 1, 12 + 1, endpoint=True)
ages = numpy.linspace(0, 20, 301, endpoint=True)

def get_age_structure(ages, start_time):
    parameters = herd.Parameters()
    parameters.start_time = start_time
    return herd.age_structure.gen(parameters).pdf(ages)

# This seems to be faster sequentially...
with Parallel(n_jobs=1) as parallel:
    age_structures = parallel(delayed(get_age_structure)(ages, start_time)
                              for start_time in start_times)

fig, ax = pyplot.subplots()
im = ax.imshow(age_structures,
               extent=(ages[0], ages[-1], start_times[0], start_times[-1]),
               cmap='viridis', interpolation='bilinear',
               origin='bottom', aspect='auto')
ax.set_xlabel('age (y)')
ax.set_ylabel('start time (y)')
fig.colorbar(im, label='density (y$^{-1}$)')
fig.tight_layout()
pyplot.show()
