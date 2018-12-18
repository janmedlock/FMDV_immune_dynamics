#!/usr/bin/python3
import subprocess
from time import perf_counter, process_time

from matplotlib import pyplot
import numpy
from scipy.integrate import quad

import sys
sys.path.append('..')

# Rebuild the 'monodromy' module, if necessary.
subprocess.run(['make'], cwd='../herd/floquet', check=True)

from herd import mortality, Parameters
from herd.floquet import floquet
from herd.utility import arange


def find_stable_age_structure(parameters, fast=False):
    solver_parameters = floquet.monodromy.Parameters(parameters)
    # Temporarily monkeypatch non-caching version of
    # `_find_dominant_eigen` into place to force recomputation.
    _find_dominant_eigen = floquet._find_dominant_eigen
    floquet._find_dominant_eigen = _find_dominant_eigen.func
    try:
        t0, p0 = perf_counter(), process_time()
        if fast:
            birth_scaling = 0.9378975738425385
            r, v, ages = floquet._find_dominant_eigen(birth_scaling,
                                                      solver_parameters,
                                                      floquet._agemax_default,
                                                      floquet._agestep_default)
        else:
            # Force a recomputation of the birth scaling.
            birth_scaling = floquet._find_birth_scaling.func(
                solver_parameters,
                floquet._agemax_default,
                floquet._agestep_default)
        t1, p1 = perf_counter(), process_time()
    finally:
        # Restore caching version of `_find_dominant_eigen`.
        floquet._find_dominant_eigen = _find_dominant_eigen
    print('{} benchmark took {:g} s clock, {:g} s CPU'.format(
        'Fast' if fast else 'Slow', t1 - t0, p1 - p0))
    if fast:
        assert numpy.isclose(r, 0), 'r = {:g}'.format(r)
    else:
        v, ages = floquet.find_stable_age_structure(parameters)
    return (v, ages)


def plot(ages, stable_age_structure):
    fig, ax = pyplot.subplots()
    agemax = 25
    which = (ages <= agemax)
    ax.plot(ages[which], stable_age_structure[which],
            label='stable age structure')
    mortality_sf = mortality.from_param_values().sf
    mortality_sf_scale, _ = quad(mortality_sf, ages[0], ages[-1])
    ax.plot(ages[which], mortality_sf(ages[which]) / mortality_sf_scale,
            label='scaled mortality survival',
            color='black', linestyle='dotted')
    ax.set_xlabel('age (y)')
    ax.set_ylabel('density (y$^{-1}$)')
    ax.legend()
    fig.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    parameters = Parameters()
    stable_age_structure, ages = find_stable_age_structure(parameters,
                                                           fast=True)
    plot(ages, stable_age_structure)
