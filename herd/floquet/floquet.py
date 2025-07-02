import pathlib

from joblib import Memory
import numpy
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from herd import parameters, periods
from herd.floquet import dominant_eigen, monodromy


_step_default = 0.01
_age_max_default = 35


def _normalize_to_density(v, ages):
    '''Normalize `v` in place so that its integral over ages is 1.'''
    v /= trapezoid(v, ages)


# Some of the functions below are very slow, so the values are cached
# to disk with `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cache_path = pathlib.Path(__file__).with_name('_cache')
_cache = Memory(_cache_path, verbose=1)


# This function is very slow because it calls
# `monodromy.Solver.solve()`, which is very slow, and
# `dominant_eigen.find()`, which is somewhat slow, so it is cached.
# Caching also allows the eigenvector to be retrieved from the cache
# by `find_stable_age_structure()` after the eigenvalue is computed
# in `find_birth_scaling()`.
# Set `verbose=0` because this function gets called many times in
# `_find_birth_scaling()`, which leads to lots of output if
# `verbose>0`.  `_find_birth_scaling()` is cached and currently the
# default for the cache has `verbose>0`, so timing info etc is shown
# there.
@_cache.cache(ignore=['solver'], verbose=0)
def _find_dominant_eigen(birth_scaling, params,
                         step, age_max, solver=None):
    '''Find the dominant Floquet exponent (the one with the largest real part)
    and its corresponding eigenvector.'''
    if solver is None:
        solver = monodromy.Solver(params, step, age_max)
    PhiT = solver.solve(birth_scaling)
    # Finding the matrix B = log(Phi(T)) / T is very expensive,
    # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
    # and convert.
    rho0, v0 = dominant_eigen.find(PhiT, which='LM')
    # rho0 is the dominant (largest magnitude) Floquet multiplier.
    # mu0 is the dominant (largest real part) Floquet exponent.
    # They are related by rho0 = exp(mu0 * T).
    mu0 = numpy.log(rho0) / periods.get_period()
    # v0 is the eigenvector for both rho0 and mu0.
    _normalize_to_density(v0, solver.ages)
    return (mu0, v0, solver.ages)


# This function is not cached because it just calls
# `_find_dominant_eigen()`, which is.
def _find_growth_rate(birth_scaling, params,
                      step, age_max, solver):
    '''Find the population growth rate.'''
    mu0, _, _ = _find_dominant_eigen(birth_scaling, params,
                                     step, age_max, solver)
    return mu0


# This function is extremely slow because, through
# `_find_growth_rate()` and `scipy.optimize.brentq()`, it repeatedly
# calls `_find_dominant_eigen()`, which is very slow, so it is cached.
# Even though `_find_dominant_eigen()` is cached, caching this function
# avoids re-running the optimization step.
@_cache.cache
def _find_birth_scaling(params, step, age_max):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Reuse the solver to avoid multiple setup/teardown.
    solver = monodromy.Solver(params, step, age_max)
    args = (params, step, age_max, solver)
    a = 0
    # We know that at the lower limit a = 0,
    # `_find_growth_rate(0, ...) < 0`,
    # so we need to find an upper limit `b`
    # with `_find_growth_rate(b, ...) >= 0`.
    b = 1
    while _find_growth_rate(b, *args) < 0:
        a = b
        b *= 2
    (birth_scaling, res) = brentq(_find_growth_rate, a, b,
                                  args=args,
                                  full_output=True)
    assert res.converged, res
    return birth_scaling


class _CacheParameters(parameters.Parameters):
    '''Build a `herd.parameters.Parameters()`-like object that
    only has the parameters needed by `Solver()`
    so that it can be efficiently cached.'''
    def __init__(self, params):
        # Generally, the values of these parameters should be
        # floats, so explicitly convert them so the cache doesn't
        # get duplicated keys for the float and int representation
        # of the same number, e.g. `float(0)` and `int(0)`.
        self.birth_seasonal_coefficient_of_variation = float(
            params.birth_seasonal_coefficient_of_variation)
        # Normalize `params.birth_peak_time_of_year` by making
        # it relative to `params.start_time` and then modulo
        # `period` so that it is in [0, period).
        self.birth_peak_time_of_year = float(
            (params.birth_peak_time_of_year - params.start_time)
            % periods.get_period())
        self.start_time = float(0)
        self.female_probability_at_birth = float(
            params.female_probability_at_birth)


# Wrapper to call cached version, `_find_birth_scaling()`.
def find_birth_scaling(params,
                       step=_step_default,
                       age_max=_age_max_default):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    return _find_birth_scaling(_CacheParameters(params), step, age_max)


# Because `find_birth_scaling()` fills the cache for
# `_find_dominant_eigen()`, the eigenvector needed, `v`, should
# already be in the cache, and thus, there is no need to cache this
# function.
def find_stable_age_structure(params,
                              step=_step_default,
                              age_max=_age_max_default,
                              _birth_scaling=None):
    '''Find the stable age structure.'''
    cache_params = _CacheParameters(params)
    if _birth_scaling is None:
        _birth_scaling = _find_birth_scaling(cache_params,
                                             step, age_max)
    r, v, ages = _find_dominant_eigen(_birth_scaling, cache_params,
                                      step, age_max)
    assert numpy.isclose(r, 0), 'Nonzero growth rate r={:g}.'.format(r)
    return (v, ages)
