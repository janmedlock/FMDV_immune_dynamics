#!/usr/bin/env python
'''Time the Floquet solver.'''

import argparse

import matplotlib.pyplot
import scipy

from _context import common, herd
import herd._floquet.test

AGE_MAX = 25


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--slow', dest='fast', action='store_false')
    return parser.parse_args()


def plot(parameters, ages, stable_age_structure):

    def mortality_sf(age):
        return herd.events.mortality.sf(
            parameters.mortality_age_dependent, age)

    (fig, ax) = matplotlib.pyplot.subplots()
    which = (ages <= AGE_MAX)
    ax.plot(ages[which], stable_age_structure[which],
            label='stable age structure')
    (mortality_sf_scale, _) = scipy.integrate.quad(mortality_sf,
                                                   ages[0], ages[-1])
    ax.plot(ages[which],
            mortality_sf(ages[which]) / mortality_sf_scale,
            label='scaled mortality survival',
            color='black', linestyle='dotted')
    ax.set_xlabel(f'age ({common.TIME_UNIT})')
    ax.set_ylabel(f'density ({common.TIME_UNIT}$^{-1}$)')
    ax.legend()
    fig.tight_layout()


if __name__ == '__main__':
    args = _parse_args()
    parameters = herd.Parameters()
    (age_structure, ages) = herd._floquet.test.find_stable_age_structure(
        parameters, fast=args.fast)
    plot(parameters, ages, age_structure)
    matplotlib.pyplot.show()
