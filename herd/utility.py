import shelve
import inspect
import os.path

import numpy
from scipy import integrate, optimize, sparse


class _BirthMortalityParameters:
    '''Container to extract birth & mortality parameters from `parameters`.'''
    _attrs = ('birth_seasonal_coefficient_of_variation',
              'male_probability_at_birth')
    def __init__(self, parameters):
        for a in self._attrs:
            setattr(self, a, getattr(parameters, a))
    def __repr__(self):
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramreprs = ['{!r}: {!r}'.format(a, getattr(self, a))
                      for a in self._attrs]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))


class _shelved:
    '''Decorator to memoize results and store to disk.
    The cache key is repr(parameters).'''
    def __init__(self, func):
        self.func = func
        # Put the cache file in the same directory as the caller.
        mydir = os.path.dirname(inspect.getfile(self.func))
        # Name the cache file func.__name__ + '.db'
        self.myfile = os.path.join(mydir,
                                   str(self.func.__name__))

    def __call__(self, parameters, *args, **kwargs):
        key = repr(parameters)
        cache = shelve.open(self.myfile)
        try:
            val = cache[key]
        except (KeyError, ValueError, TypeError):
            print('{} not in {} cache.  Computing...'.format(
                key, self.func.__name__))
            val = cache[key] = self.func(parameters, *args, **kwargs)
            print('\tFinished computing {}.'.format(self.func.__name__))
        finally:
            cache.close()
        return val


def _build_matrices(parameters, agemax=20, agestep=0.01):
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    da = numpy.diff(ages)
    # Aging
    A = sparse.lil_matrix((len(ages), ) * 2)
    A.setdiag(- numpy.hstack((1 / da, 1 / da[-1])))
    A.setdiag(1 / da, -1)
    # Mortality
    from . import mortality
    mortalityRV = mortality.gen(parameters)
    M = sparse.lil_matrix((len(ages), ) * 2)
    M.setdiag(mortalityRV.hazard(ages))
    # Birth
    from . import birth
    birthRV = birth.gen(parameters, _find_birth_scaling=False)
    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    def b(j):
        def bj(t):
            return ((1 - parameters.male_probability_at_birth)
                    * birthRV.hazard(t, 0, ages[j] - t))
        return bj
    for j in range(len(ages)):
        B_bar[0, j], _ = integrate.quadrature(b(j), 0, 1, maxiter=1000)
    return (ages, (B_bar, A, M))


def _find_dominant_eigenpair(parameters,
                             _birth_scaling=1, _matrices=None,
                             *args, **kwargs):
    if _matrices is None:
        _matrices = _build_matrices(parameters,
                                    *args, **kwargs)
    (ages, (B_bar, A, M)) = _matrices
    G = _birth_scaling * B_bar + A - M
    # Find the largest real eigenvalue.
    [L, V] = sparse.linalg.eigs(G, k=1, which='LR',
                                maxiter=int(1e5))
    l0 = numpy.asscalar(numpy.real_if_close(L[0]))
    v0 = numpy.real_if_close(V[:, 0] / numpy.sum(V[:, 0]))
    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)
    return (l0, (ages, v0))


def _find_growth_rate(parameters, *args, **kwargs):
    r, _ = _find_dominant_eigenpair(parameters, *args, **kwargs)
    return r


@_shelved
def _find_stable_age_structure(parameters,
                               _matrices=None, _birth_scaling=None,
                               *args, **kwargs):
    if _matrices is None:
        _matrices = _build_matrices(parameters, *args, **kwargs)
    if _birth_scaling is None:
        _birth_scaling = _find_birth_scaling(parameters,
                                             _matrices=_matrices,
                                             *args, **kwargs)
    _, v = _find_dominant_eigenpair(parameters,
                                    _birth_scaling=_birth_scaling,
                                    _matrices=_matrices,
                                    *args, **kwargs)
    return v

def find_stable_age_structure(parameters, *args, **kwargs):
    return _find_stable_age_structure(_BirthMortalityParameters(parameters),
                                      *args, **kwargs)


@_shelved
def _find_birth_scaling(parameters,
                        _birth_scaling=1, _matrices=None,
                        *args, **kwargs):
    if _matrices is None:
        _matrices = _build_matrices(parameters, *args, **kwargs)
    def _objective(z):
        return _find_growth_rate(parameters,
                                 _birth_scaling=numpy.asscalar(z),
                                 _matrices=_matrices,
                                 *args, **kwargs)
    birth_scaling, _, ier, mesg = optimize.fsolve(_objective, _birth_scaling,
                                                  full_output=True)
    assert ier == 1, mesg
    return numpy.asscalar(birth_scaling)

def find_birth_scaling(parameters, *args, **kwargs):
    return _find_birth_scaling(_BirthMortalityParameters(parameters),
                               *args, **kwargs)
