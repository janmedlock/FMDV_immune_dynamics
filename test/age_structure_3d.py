#!/usr/bin/env python
'''Plot the stable age distribution.'''

import matplotlib.pyplot
import numpy

from _context import common

import age_structure

birth_peak_times = numpy.linspace(0, 1, 12 + 1, endpoint=True)
ages = numpy.linspace(0, 20, 301, endpoint=True)


def plot_age_structures_3d(birth_peak_times, age_structures):
    (fig, ax) = matplotlib.pyplot.subplots()
    pcm = ax.pcolormesh(ages, birth_peak_times, age_structures,
                        cmap='viridis', shading='gouraud')
    ax.set_xlabel(f'age ({common.TIME_UNIT})')
    ax.set_ylabel(f'start time ({common.TIME_UNIT})')
    fig.colorbar(pcm, label=f'density ({common.TIME_UNIT}$^{{-1}}$)')
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    age_structures = age_structure.get_age_structures(ages, birth_peak_times)
    plot_age_structures_3d(birth_peak_times, age_structures)
    matplotlib.pyplot.show()
