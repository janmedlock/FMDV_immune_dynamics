#!/usr/bin/python3
'''Plot the stable age distribution.'''

import numpy
from matplotlib import pyplot

from context import herd
import herd.age_structure
import age_structure


start_times = numpy.linspace(0, 1, 12 + 1, endpoint=True)
ages = numpy.linspace(0, 20, 301, endpoint=True)


def plot_age_structures_3d(age_structures, show=True):
    (fig, ax) = pyplot.subplots()
    pcm = ax.pcolormesh(ages, start_times, age_structures,
                        cmap='viridis', shading='gouraud')
    ax.set_xlabel('age (y)')
    ax.set_ylabel('start time (y)')
    fig.colorbar(pcm, label='density (y$^{-1}$)')
    fig.tight_layout()
    if show:
        pyplot.show()
    return ax


if __name__ == '__main__':
    age_structures = age_structure.get_age_structures(ages, start_times)
    plot_age_structures_3d(age_structures)
