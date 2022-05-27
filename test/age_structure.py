#!/usr/bin/python3
from joblib import delayed, Parallel
from matplotlib import pyplot
import numpy
from scipy.integrate import quad

from context import herd
import herd.age_structure
import herd.mortality
import herd.utility.arange

start_times = numpy.linspace(0, 1, 4, endpoint=False)
ages = herd.utility.arange(0, 25, 0.01, endpoint=True)

def get_age_structure(ages, start_time):
    parameters = herd.Parameters()
    parameters.start_time = start_time
    return herd.age_structure.gen(parameters).pdf(ages)

# This seems to be faster sequentially...
with Parallel(n_jobs=1) as parallel:
    age_structures = parallel(delayed(get_age_structure)(ages, start_time)
                              for start_time in start_times)

mortality_sf = herd.mortality.from_param_values().sf
mortality_sf_scale, _ = quad(mortality_sf, ages[0], ages[-1])

fig, ax = pyplot.subplots()
for (start_time, age_structure) in zip(start_times, age_structures):
    ax.plot(ages, age_structure, label='{:g} months'.format(12 * start_time),
            alpha=0.7)
ax.plot(ages, mortality_sf(ages) / mortality_sf_scale,
        label='scaled mortality survival',
        color='black', linestyle='dotted')
ax.set_xlabel('age (y)')
ax.set_ylabel('density (y$^{-1}$)')
ax.legend(title='start time')
fig.tight_layout()
pyplot.show()
