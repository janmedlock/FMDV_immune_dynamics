#!/usr/bin/env python
'''Time the Floquet solver.'''

import time

import matplotlib.pyplot
import numpy
import scipy

from _context import common, herd
import herd._floquet.floquet
import herd.events.mortality


def find_stable_age_structure(parameters, fast=False):
    cache_parameters = herd._floquet.floquet._CacheParameters(parameters)
    # Temporarily monkeypatch non-caching version of
    # `_find_dominant_eigen` into place to force recomputation.
    _find_dominant_eigen = herd._floquet.floquet._find_dominant_eigen
    herd._floquet.floquet._find_dominant_eigen = _find_dominant_eigen.func
    try:
        t0, p0 = time.perf_counter(), time.process_time()
        if fast:
            birth_scaling = 0.9390954290776172
            r, v, ages = herd._floquet.floquet._find_dominant_eigen(
                birth_scaling, cache_parameters)
        else:
            # Force a recomputation of the birth scaling.
            birth_scaling = herd._floquet.floquet._find_birth_scaling.func(
                cache_parameters)
        t1, p1 = time.perf_counter(), time.process_time()
    finally:
        # Restore caching version of `_find_dominant_eigen`.
        herd._floquet.floquet._find_dominant_eigen = _find_dominant_eigen
    print('{} benchmark took {:g} s clock, {:g} s CPU'.format(
        'Fast' if fast else 'Slow', t1 - t0, p1 - p0))
    if fast:
        # assert numpy.isclose(r, 0), 'r = {:g}'.format(r)
        if not numpy.isclose(r, 0):
            print(f'r = {r:g}')
    else:
        v, ages = herd._floquet.find_stable_age_structure(parameters)
    return (v, ages)


def plot(parameters, ages, stable_age_structure):

    def mortality_sf(age):
        return herd.events.mortality.sf(parameters.mortality_age_dependent,
                                        age)

    fig, ax = matplotlib.pyplot.subplots()
    age_max = 25
    which = (ages <= age_max)
    ax.plot(ages[which], stable_age_structure[which],
            label='stable age structure')
    mortality_sf_scale, _ = scipy.integrate.quad(mortality_sf,
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
    FAST = True
    parameters = herd.Parameters()
    stable_age_structure, ages = find_stable_age_structure(parameters,
                                                           fast=FAST)
    plot(parameters, ages, stable_age_structure)
    matplotlib.pyplot.show()
