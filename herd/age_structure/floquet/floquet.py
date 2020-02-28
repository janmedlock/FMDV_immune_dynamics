import os.path

from joblib import Memory
import numpy
from scipy.integrate import trapz
from scipy.optimize import brentq

from herd.age_structure.floquet import dominant_eigen, monodromy
from herd.age_structure.floquet import period


_agemax_default = 35
_agestep_default = 0.01


def _normalize_to_density(v, ages):
    '''Normalize `v` in place so that its integral over ages is 1.'''
    v /= trapz(v, ages)


# Some of the functions below are very slow, so the values are cached
# to disk with `joblib.Memory()` so that they are only computed once.
# Set up the cache in a subdirectory of the directory that this source
# file is in.
_cachedir = os.path.join(os.path.dirname(__file__), '_cache')
_cache = Memory(_cachedir, verbose=1)


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
def _find_dominant_eigen(birth_scaling, solver_parameters,
                         agemax, agestep, solver=None):
    '''Find the dominant Floquet exponent (the one with the largest real part)
    and its corresponding eigenvector.'''
    if solver is None:
        solver = monodromy.Solver(solver_parameters, agemax, agestep)
    PhiT = solver.solve(birth_scaling)
    # Finding the matrix B = log(Phi(T)) / T is very expensive,
    # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
    # and convert.
    rho0, v0 = dominant_eigen.find(PhiT, which='LM')
    # rho0 is the dominant (largest magnitude) Floquet multiplier.
    # mu0 is the dominant (largest real part) Floquet exponent.
    # They are related by rho0 = exp(mu0 * T).
    mu0 = numpy.log(rho0) / period.get()
    # v0 is the eigenvector for both rho0 and mu0.
    _normalize_to_density(v0, solver.ages)
    return (mu0, v0, solver.ages)


# This function is not cached because it just calls
# `_find_dominant_eigen()`, which is.
def _find_growth_rate(birth_scaling, solver_parameters,
                      agemax, agestep, solver):
    '''Find the population growth rate.'''
    mu0, _, _ = _find_dominant_eigen(birth_scaling, solver_parameters,
                                     agemax, agestep, solver)
    return mu0


# This function is extremely slow because, through
# `_find_growth_rate()` and `scipy.optimize.brentq()`, it repeatedly
# calls `_find_dominant_eigen()`, which is very slow, so it is cached.
# Even though `_find_dominant_eigen()` is cached, caching this function
# avoids re-running the optimization step.
@_cache.cache
def _find_birth_scaling(solver_parameters, agemax, agestep):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Reuse the solver to avoid multiple setup/teardown.
    solver = monodromy.Solver(solver_parameters, agemax, agestep)
    args = (solver_parameters, agemax, agestep, solver)
    a = 0
    # We know that at the lower limit a = 0,
    # `_find_growth_rate(0, ...) < 0`,
    # so we need to find an upper limit `b`
    # with `_find_growth_rate(b, ...) >= 0`.
    b = 1
    while _find_growth_rate(b, *args) < 0:
        a = b
        b *= 2
    return brentq(_find_growth_rate, a, b, args=args)


# Wrapper to call cached version, `_find_birth_scaling()`.
def find_birth_scaling(parameters,
                       agemax=_agemax_default,
                       agestep=_agestep_default):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    solver_parameters = monodromy.Parameters(parameters)
    return _find_birth_scaling(solver_parameters, agemax, agestep)


# Because `find_birth_scaling()` fills the cache for
# `_find_dominant_eigen()`, the eigenvector needed, `v`, should
# already be in the cache, and thus, there is no need to cache this
# function.
def find_stable_age_structure(parameters,
                              agemax=_agemax_default,
                              agestep=_agestep_default,
                              _birth_scaling=None):
    '''Find the stable age structure.'''
    solver_parameters = monodromy.Parameters(parameters)
    if _birth_scaling is None:
        _birth_scaling = _find_birth_scaling(solver_parameters,
                                             agemax, agestep)
    r, v, ages = _find_dominant_eigen(_birth_scaling, solver_parameters,
                                      agemax, agestep)
    assert numpy.isclose(r, 0), 'Nonzero growth rate r={:g}.'.format(r)
    return (v, ages)
