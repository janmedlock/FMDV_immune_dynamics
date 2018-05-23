import functools
import inspect
import os.path
import shelve

import numpy
from scipy import optimize, sparse


class _shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''
    def __init__(self, *parameters_to_keep):
        self._parameters_to_keep = parameters_to_keep

    def _get_key(self, parameters):
        clsname = '{}.{}'.format(parameters.__module__,
                                 parameters.__class__.__name__)
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self._parameters_to_keep)
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    def __call__(self, func):
        # Put the cache file in the same directory as the caller
        # and name it 'module.func.db'.
        root, _ = os.path.splitext(inspect.getfile(func))
        cachefile = '{}.{}'.format(root, func.__name__)
        @functools.wraps(func)
        def wrapped(parameters, *args, **kwargs):
            cache = shelve.open(cachefile)
            key = self._get_key(parameters)
            try:
                val = cache[key]
            except (KeyError, ValueError, TypeError):
                print('{} not in {} cache.  Computing...'.format(key,
                                                                 func.__name__))
                val = cache[key] = func(parameters, *args, **kwargs)
                print('\tFinished computing {}.'.format(func.__name__))
            finally:
                cache.close()
            return val
        return wrapped


def _build_ages_and_matrices(parameters, agemax=25, agestep=0.01):
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    # Aging
    A = sparse.lil_matrix((len(ages), ) * 2)
    da = numpy.diff(ages)
    A.setdiag(- numpy.hstack((1 / da, 0)))
    A.setdiag(1 / da, -1)
    # Mortality
    M = sparse.lil_matrix((len(ages), ) * 2)
    from . import mortality
    mortalityRV = mortality.gen(parameters)
    M.setdiag(mortalityRV.hazard(ages))
    # Birth
    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    from . import birth
    birthRV = birth.gen(parameters, _find_birth_scaling=False)
    B_bar[0] = ((1 - parameters.male_probability_at_birth)
                * (- birthRV.logsf(1, parameters.start_time, ages)))
    return (ages, (B_bar, A, M))


def _find_dominant_eigenpair(birth_scaling, B_bar, A, M):
    G = birth_scaling * B_bar + A - M
    # Find the largest real eigenvalue.
    L, V = sparse.linalg.eigs(G, k=1, which='LR', maxiter=int(1e5))
    V /= V.sum(axis=0)
    l0, v0 = map(numpy.squeeze, (L, V))
    l0, v0 = map(numpy.real_if_close, (l0, v0))
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    return (l0, v0)


# `start_time` doesn't matter since we're integrating a 1-year-periodic
# function over 1 year.
@_shelved('birth_seasonal_coefficient_of_variation',
          'male_probability_at_birth')
def find_birth_scaling(parameters, _matrices=None, *args, **kwargs):
    if _matrices is None:
        _, _matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    def _objective(val, *matrices):
        birth_scaling, = val
        r, _ = _find_dominant_eigenpair(birth_scaling, *matrices)
        return r
    initial_guess = 1
    opt, _, ier, mesg = optimize.fsolve(_objective, initial_guess,
                                        args=_matrices,
                                        full_output=True)
    birth_scaling, = opt
    assert ier == 1, mesg
    return birth_scaling


@_shelved('birth_seasonal_coefficient_of_variation',
          'male_probability_at_birth',
          'start_time')
def find_stable_age_structure(parameters, *args, **kwargs):
    ages, matrices = _build_ages_and_matrices(parameters, *args, **kwargs)
    birth_scaling = find_birth_scaling(parameters, _matrices=matrices)
    _, v = _find_dominant_eigenpair(birth_scaling, *matrices)
    return (ages, v)
