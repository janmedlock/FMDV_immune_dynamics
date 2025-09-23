#!/usr/bin/python3
'''Plot the stable age distribution.'''

import joblib
import matplotlib.pyplot
import numpy
import scipy

from context import common
from context import herd
import herd.age_structure
import herd.mortality
import herd.utility


start_times = numpy.linspace(0, 1, 4, endpoint=False)
ages = herd.utility.arange(0, 25, 0.01, endpoint=True)
N_JOBS = 1  # This seems to be faster sequentially.


def get_age_structure(ages, start_time):
    parameters = herd.Parameters()
    parameters.start_time = start_time
    return herd.age_structure.gen(parameters).pdf(ages)


def get_age_structures(ages, start_times):
    with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        return parallel(
            joblib.delayed(get_age_structure)(ages, start_time)
            for start_time in start_times
        )


def plot_age_structures(age_structures):
    mortality_sf_scale, _ = scipy.integrate.quad(
        herd.mortality.sf, ages[0], ages[-1]
    )
    (fig, ax) = matplotlib.pyplot.subplots()
    for (start_time, age_structure) in zip(start_times, age_structures):
        ax.plot(ages, age_structure,
                label='{:g} months'.format(12 * start_time),
                alpha=0.7)
    ax.plot(ages, herd.mortality.sf(ages) / mortality_sf_scale,
            label='scaled mortality survival',
            color='black', linestyle='dotted')
    ax.set_xlabel(f'age ({common.TIME_UNIT})')
    ax.set_ylabel('density (y$^{-1}$)')
    ax.legend(title='start time')
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    age_structures = get_age_structures(ages, start_times)
    plot_age_structures(age_structures)
    matplotlib.pyplot.show()
