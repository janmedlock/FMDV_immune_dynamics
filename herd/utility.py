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
    mean_birth_rate = ((1 - parameters.male_probability_at_birth)
                       * cumulative_hazard)
    # The index values of the first row (0, j).
    indexes = (numpy.zeros(N), numpy.arange(N))
    # Use CSR for fast multiply.
    B_bar = sparse.csr_matrix((mean_birth_rate, indexes), shape=(N, N))
    # Mortality and aging
    mortalityRV = mortality.gen(parameters)
    mortality_rate = mortalityRV.hazard(ages)
    # Don't fall out of the last age group.
    aging_rate = numpy.hstack((1 / numpy.diff(ages), 0))
    offsets, diags = zip((0, - mortality_rate - aging_rate), # Diagonal, offset 0.
                         (-1, aging_rate[: -1]))         # Subdiagonal, offset -1.
    # Use CSR for consistency with B_bar.
    MA = sparse.diags(diags, offsets, format='csr')
    return (ages, (B_bar, MA))


def find_dominant_eigenpair(birth_scaling, B_bar, MA):
    G = birth_scaling * B_bar + MA
    # Find the eigenvalue with largest real part.
    L, V = sparse.linalg.eigs(G, k=1, which='LR', maxiter=100000)
    V /= V.sum(axis=0)
    l0, v0 = map(numpy.squeeze, (L, V))
    l0, v0 = map(numpy.real_if_close, (l0, v0))
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
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
