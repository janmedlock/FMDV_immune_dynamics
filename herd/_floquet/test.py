import contextlib
import functools
import math
import time

from . import _monodromy, floquet


def _time_uncached(func):
    '''Decorator for timing.'''

    @functools.wraps(func)
    def wrapper(*args, **kwds):
        (t_0, p_0) = (time.perf_counter(), time.process_time())
        # Use the uncached `.func()` to force computation
        result = func.func(*args, **kwds)
        (t, p) = (time.perf_counter() - t_0, time.process_time() - p_0)
        return (result, t, p)

    return wrapper


def _find_stable_age_structure_fast(monodromy_params):
    '''Compute the dominant eigenvalue and eigenvector.'''
    solver = _monodromy.Solver(monodromy_params)
    # Don't time `floquet._find_birth_scaling()`. Hopefully the result
    # is in the cache.
    birth_scaling = floquet._find_birth_scaling(monodromy_params,
                                                solver=solver)
    ((r, v, a), t, p) = _time_uncached(floquet._find_dominant_eigen)(
        birth_scaling, monodromy_params, solver=solver
    )
    if not math.isclose(r, 0):
        print(f'{r=:g}')
    return (v, a, t, p)


@contextlib.contextmanager
def _uncached_find_dominant_eigen():
    '''Temporarily monkeypatch `_find_dominant_eigen()` to its
    non-caching version.'''
    _find_dominant_eigen = floquet._find_dominant_eigen
    floquet._find_dominant_eigen = _find_dominant_eigen.func
    try:
        yield
    finally:
        floquet._find_dominant_eigen = _find_dominant_eigen


def _find_stable_age_structure_slow(monodromy_params):
    '''Compute the birth scaling.'''
    with _uncached_find_dominant_eigen():
        (_, t, p) = _time_uncached(floquet._find_birth_scaling)(
            monodromy_params
        )
    (v, a) = floquet.find_stable_age_structure(monodromy_params)
    return (v, a, t, p)


def find_stable_age_structure(params, fast=True):
    '''Benchmark `floquet`.'''
    find_stable_age_structure_ = (
        _find_stable_age_structure_fast if fast
        else _find_stable_age_structure_slow
    )
    monodromy_params = _monodromy.Parameters(params)
    (v, a, t, p) = find_stable_age_structure_(monodromy_params)
    fast_str = 'Fast' if fast else 'Slow'
    print(f'{fast_str} benchmark took {t:g} s clock, {p:g} s CPU')
    return (v, a)
