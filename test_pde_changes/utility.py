import shelve
import inspect
import os.path

import numpy
from scipy import sparse, integrate, optimize


class shelved:
    '''
    Memoize results and store to disk.

    This is set up to be used as a decorator:
    @shelved
    def func(parameters, *args, **kwargs):
        ...

    The cache key is repr(parameters).
    '''    

    def __init__(self, func):
        self.func = func
        
        # Put the cache file in the same directory as the caller.
        mydir = os.path.dirname(inspect.getfile(self.func))
        # Name the cache file func.__name__ + '.db'
        self.myfile = os.path.join(mydir,
                                   str(self.func.__name__))

    def __call__(self, parameters, *args, **kwargs):
        # Derive the shelve key from the parameters object.
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


def build_matrices(parameters, agemax = 20, agestep = 0.01):
    import mortality
    import birth
    print("building matricies")
    ages = numpy.arange(0, agemax + agestep / 2, agestep)
    da = numpy.diff(ages)

    # Aging
    A = sparse.lil_matrix((len(ages), ) * 2)
    A.setdiag(- numpy.hstack((1 / da, 0.)))
    A.setdiag(1 / da, -1)
    
    # Mortality
    mortalityRV = mortality.gen(parameters)
    M = sparse.lil_matrix((len(ages), ) * 2)
    M.setdiag(mortalityRV.hazard(ages))
    
    # Birth
    birthRV = birth.gen(parameters, _find_birth_scaling = False)
    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in range(len(ages)):
        bj = lambda t: ((1 - parameters.male_probability_at_birth)
                        * birthRV.hazard(t, 0, ages[j] - t))
        result = integrate.quad(bj, 0, 1, limit = 100)
        B_bar[0, j] = result[0]


    return (ages, (B_bar, A, M))


def find_dominant_eigenpair(parameters,
                            _birth_scaling = 1, _matrices = None,
                            *args, **kwargs):
    if _matrices is None:
        _matrices = build_matrices(parameters, *args, **kwargs)

    (ages, (B_bar, A, M)) = _matrices

    G = _birth_scaling * B_bar + A - M

    # Find the largest real eigenvalue.
    [L, V] = sparse.linalg.eigs(G, k = 1, which = 'LR',
                                maxiter = int(1e5))

    l0 = numpy.asscalar(numpy.real_if_close(L[0]))
    v0 = numpy.real_if_close(V[:, 0] / numpy.sum(V[:, 0]))

    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)

    return (l0, (ages, v0))


def find_growth_rate(parameters, *args, **kwargs):
    return find_dominant_eigenpair(parameters, *args, **kwargs)[0]


@shelved
def _find_stable_age_structure(parameters, *args, **kwargs):
    return find_dominant_eigenpair(parameters, *args, **kwargs)[1]

def find_stable_age_structure(parameters, *args, **kwargs):
    # stable age structure is independent of population_size and
    # start_time, so factor them out for more efficient caching.
    population_size = parameters.population_size
    start_time = parameters.start_time
    del parameters.population_size
    del parameters.start_time
    SAS = _find_stable_age_structure(parameters, *args, **kwargs)
    parameters.population_size = population_size
    parameters.start_time = start_time
    return SAS


@shelved
def _find_birth_scaling(parameters, *args, **kwargs):
    matrices = build_matrices(parameters, *args, **kwargs)

    def _objective(z):
        return find_growth_rate(parameters,
                                _birth_scaling = numpy.asscalar(z),
                                _matrices = matrices,
                                *args, **kwargs)

    scaling = numpy.asscalar(optimize.fsolve(_objective, 0.443))

    return scaling

def find_birth_scaling(parameters, *args, **kwargs):
    # birthScaling is independent of population_size and start_time so
    # factor then out for more efficient caching.
    population_size = parameters.population_size
    start_time = parameters.start_time
    del parameters.population_size
    del parameters.start_time
    scaling = _find_birth_scaling(parameters, *args, **kwargs)
    parameters.population_size = population_size
    parameters.start_time = start_time
    return scaling
