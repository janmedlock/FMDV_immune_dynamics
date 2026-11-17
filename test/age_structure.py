#!/usr/bin/env python
'''Plot the stable age distribution.'''

import joblib
import matplotlib.pyplot
import numpy
import scipy

from _context import common, herd
import herd.age
import herd.utility

birth_peak_times = numpy.linspace(0, 1, 4, endpoint=False)
ages = herd.utility.arange(0, 25, 0.01, endpoint=True)
N_JOBS = 1  # This seems to be faster sequentially.


def get_age_structure(ages, birth_peak_time):
    parameters = herd.Parameters()
    parameters.birth_peak_time = birth_peak_time
    return herd.age.Age(parameters).pdf(ages)


def get_age_structures(ages, birth_peak_times):
    with joblib.Parallel(n_jobs=N_JOBS) as parallel:
        return parallel(
            joblib.delayed(get_age_structure)(ages, birth_peak_time)
            for birth_peak_time in birth_peak_times
        )


def plot_age_structures(birth_peak_times, age_structures):
    parameters = herd.Parameters()

    def mortality_sf(age):
        return herd.events.mortality.sf(parameters.mortality_age_dependent,
                                        age)

    mortality_sf_scale, _ = scipy.integrate.quad(mortality_sf,
                                                 ages[0], ages[-1])
    (fig, ax) = matplotlib.pyplot.subplots()
    for (birth_peak_time, age_structure) in zip(birth_peak_times,
                                                age_structures):
        month = 12 * birth_peak_time
        ax.plot(ages, age_structure,
                label=f'{month:g} months',
                alpha=0.7)
    ax.plot(ages, mortality_sf(ages) / mortality_sf_scale,
            label='scaled mortality survival',
            color='black', linestyle='dotted')
    ax.set_xlabel(f'age ({common.TIME_UNIT})')
    ax.set_ylabel(f'density ({common.TIME_UNIT}$^{{-1}}$)')
    ax.legend(title='birth_peak_time')
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    age_structures = get_age_structures(ages, birth_peak_times)
    plot_age_structures(birth_peak_times, age_structures)
    matplotlib.pyplot.show()
