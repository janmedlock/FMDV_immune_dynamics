import math

import numpy
import scipy
import scipy.optimize.elementwise

from .. import _memory, _period
from . import _default, _dominant_eigen, _monodromy


def _check_solver(solver, monodromy_params,
                  step=_default.STEP, age_max=_default.AGE_MAX):
    assert solver.params is monodromy_params
    assert numpy.isclose(solver.step, step)
    assert numpy.isclose(solver.ages.max(),
                         math.ceil(age_max / step) * step)


def _normalize_to_density(v, step):
    '''Normalize `v` in place so that its integral over ages is 1.'''
    v /= v.sum() * step


# This function is very slow because it calls
# `_monodromy.Solver.solve()`, which is very slow, and
# `_dominant_eigen.find()`, which is somewhat slow, so it is cached.
# Caching also allows the eigenvector to be retrieved from the cache
# by `find_stable_age_structure()` after the eigenvalue is computed
# in `find_birth_scaling()`.
# Set `verbose=0` because this function gets called many times in
# `_find_birth_scaling()`, which leads to lots of output if
# `verbose>0`.  `_find_birth_scaling()` is cached and currently the
# default for the cache has `verbose>0`, so timing info etc is shown
# there.
@_memory.memory.cache(ignore=['solver'], verbose=0)
def _find_dominant_eigen(birth_scaling, monodromy_params, step=_default.STEP,
                         age_max=_default.AGE_MAX, solver=None):
    '''Find the dominant Floquet exponent (the one with the largest real part)
    and its corresponding eigenvector.'''
    if solver is None:
        solver = _monodromy.Solver(monodromy_params, step, age_max)
    else:
        _check_solver(solver, monodromy_params, step, age_max)
    PhiT = solver.solve(birth_scaling)
    # Finding the matrix B = log(Phi(T)) / T is very expensive,
    # so we'll find the dominant eigenvalue and eigenvector of Phi(T)
    # and convert.
    (rho_0, v_0) = _dominant_eigen.find(PhiT, which='LM')
    # rho_0 is the dominant (largest magnitude) Floquet multiplier.
    # mu_0 is the dominant (largest real part) Floquet exponent.
    # They are related by rho_0 = exp(mu_0 * T).
    mu_0 = numpy.log(rho_0) / _period.get_period()
    # v_0 is the eigenvector for both rho0 and mu0.
    _normalize_to_density(v_0, step)
    return (mu_0, v_0, solver.ages)


def _find_growth_rate(monodromy_params, step=_default.STEP,
                      age_max=_default.AGE_MAX, solver=None):
    '''Helper for `_find_birth_scaling()`.'''

    # This function is not cached because it just calls
    # `_find_dominant_eigen()`, which is cached.
    # `scipy.optimize.elementwise.find_root()` requires the function
    # accept arrays. We also want to limit those to only `birth_scaling`
    # so we can't use `args` in `scipy.optimize.elementwise.find_root()`
    # for `params`, `step`, `age_max`, and `solver`.
    @numpy.vectorize(otypes='d')
    def find_growth_rate(birth_scaling):
        '''The population growth rate, r.'''
        (mu_0, *_) = _find_dominant_eigen(birth_scaling, monodromy_params,
                                          step, age_max, solver)
        return mu_0

    return find_growth_rate


# `birth_scaling` must be nonnegative
_BIRTH_SCALING_MIN = 0.
# I expect it is about 1.
_BIRTH_SCALING_BRACKET_GUESS = (0.5, 1.5)


# This function is extremely slow because, through
# `_find_growth_rate()` and `scipy.optimize.elementwise.find_root()`,
# it repeatedly calls `_find_dominant_eigen()`, which is very slow, so
# it is cached.  Even though `_find_dominant_eigen()` is cached,
# caching this function avoids re-running the optimization.
@_memory.memory.cache(ignore=['solver', 'bracket_guess'])
def _find_birth_scaling(monodromy_params, step=_default.STEP,
                        age_max=_default.AGE_MAX, solver=None,
                        bracket_guess=_BIRTH_SCALING_BRACKET_GUESS):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    # Reuse the solver to avoid multiple setup and teardown.
    if solver is None:
        solver = _monodromy.Solver(monodromy_params, step, age_max)
    else:
        _check_solver(solver, monodromy_params, step, age_max)
    find_growth_rate = _find_growth_rate(monodromy_params, step, age_max,
                                         solver)
    bracket = scipy.optimize.elementwise.bracket_root(find_growth_rate,
                                                      *bracket_guess,
                                                      xmin=_BIRTH_SCALING_MIN)
    assert bracket.success, bracket
    root = scipy.optimize.elementwise.find_root(find_growth_rate,
                                                bracket.bracket)
    assert root.success, root
    return root.x


# Wrapper to call cached version, `_find_birth_scaling()`.
def find_birth_scaling(params, step=_default.STEP, age_max=_default.AGE_MAX,
                       **kwds):
    '''Find the birth scaling that gives population growth rate r = 0.'''
    return _find_birth_scaling(_monodromy.Parameters(params), step, age_max,
                               **kwds)


# Because `find_birth_scaling()` fills the cache for
# `_find_dominant_eigen()`, the eigenvector needed, `v`, should
# already be in the cache, and thus, there is no need to cache this
# function.
def find_stable_age_structure(params, step=_default.STEP,
                              age_max=_default.AGE_MAX, _birth_scaling=None):
    '''Find the stable age structure.'''
    monodromy_params = _monodromy.Parameters(params)
    if _birth_scaling is None:
        _birth_scaling = _find_birth_scaling(monodromy_params, step, age_max)
    (r, v, ages) = _find_dominant_eigen(_birth_scaling, monodromy_params,
                                        step, age_max)
    assert numpy.isclose(r, 0), f'Nonzero growth rate r={r:g}.'
    return (v, ages)
