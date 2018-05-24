import functools
import inspect
import os.path
import pickle
import shelve

import numpy
from scipy import sparse

from . import birth
from . import mortality


def build_ages_and_matrices(parameters, agemax=25, agestep=0.01):
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    N = len(ages)
    # Birth
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    # The mean integral is the cumulative hazard, which is -logsf.
    birthRV = birth.gen(parameters, _scaling=1)
    cumulative_hazard = -birthRV.logsf(1, parameters.start_time, ages)
    mean_birth_rate = parameters.female_probability_at_birth * cumulative_hazard
    B_bar = sparse.lil_matrix((N, N))
    B_bar[0] = mean_birth_rate
    # Mortality and aging
    mortalityRV = mortality.gen(parameters)
    mortality_rate = mortalityRV.hazard(ages)
    # Don't fall out of the last age group.
    aging_rate = numpy.hstack((1 / numpy.diff(ages), 0))
    T = sparse.dia_matrix((N, N))
    T.setdiag(- mortality_rate - aging_rate, 0)
    T.setdiag(aging_rate[: -1], -1)
    # Convert to CSR for fast multiply.
    matrices = [X.asformat('csr') for X in (B_bar, T)]
    return (ages, matrices)


def find_dominant_eigenpair(birth_scaling, B_bar, T):
    G = birth_scaling * B_bar + T
    # Find the eigenvalue with largest real part.
    L, V = sparse.linalg.eigs(G, k=1, which='LR', maxiter=100000)
    V /= V.sum(axis=0)
    l0, v0 = map(numpy.squeeze, (L, V))
    l0, v0 = map(numpy.real_if_close, (l0, v0))
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    assert all(numpy.isreal(v0)), 'Complex dominant eigenvector: {}'.format(v0)
    assert all(v0 >= 0), \
        'Negative component of the dominant eigenvector: {}'.format(v0)
    return (l0, v0)


class shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is derived from the `getattr(Parameters, p)`
    for `p` in `parameters_to_keep`.'''
    def __init__(self, *parameters_to_keep):
        self._parameters_to_keep = parameters_to_keep

    def get_key(self, parameters):
        clsname = '{}.{}'.format(parameters.__module__,
                                 parameters.__class__.__name__)
        paramreprs = ('{!r}: {!r}'.format(a, getattr(parameters, a))
                      for a in self._parameters_to_keep)
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))

    @staticmethod
    def get_cache_file(func):
        # Put the cache file in the same directory as the caller
        # and name it 'module.func.db'.
        root, _ = os.path.splitext(inspect.getfile(func))
        return '{}.{}'.format(root, func.__name__)

    def cached_call(self, func, parameters, *args, **kwargs):
        with shelve.open(self.get_cache_file(func),
                         protocol=pickle.HIGHEST_PROTOCOL) as shelf:
            key = self.get_key(parameters)
            try:
                val = shelf[key]
            except (KeyError, ValueError, TypeError):
                func_name = '{}.{}()'.format(func.__module__, func.__name__)
                print('{} not in {} cache.  Computing...'.format(key, func_name))
                val = shelf[key] = func(parameters, *args, **kwargs)
                print('\tFinished computing {}.'.format(func_name))
        return val

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(parameters, *args, **kwargs):
            return self.cached_call(func, parameters, *args, **kwargs)
        return wrapped
