import numpy
from scipy import sparse, integrate, optimize
import shelve
import inspect
import os.path


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
        # Name the cache file func.__name__ + '.shelve'
        self.myfile = os.path.join(mydir,
                                   '{}.shelve'.format(self.func.__name__))

    def __call__(self, parameters, *args, **kwargs):
        # Derive the shelve key from the parameters object.
        key = repr(parameters)

        cache = shelve.open(self.myfile)
        try:
            val = cache[key]
        except (KeyError, ValueError, TypeError):
            print '{} not in {} cache.  Computing...'.format(
                key, self.func.__name__)
            val = cache[key] = self.func(parameters, *args, **kwargs)
            print '\tFinished computing {}.'.format(self.func.__name__)
        finally:
            cache.close()

        return val


def buildMatrices(parameters, ageMax = 20., ageStep = 0.01):
    from . import mortality
    from . import birth

    ages = numpy.arange(0., ageMax + ageStep / 2., ageStep)
    da = numpy.diff(ages)

    # Aging
    A = sparse.lil_matrix((len(ages), ) * 2)
    A.setdiag(- numpy.hstack((1. / da, 0.)))
    A.setdiag(1. / da, -1)

    # Mortality
    mortalityRV = mortality.mortality_gen(parameters)
    M = sparse.lil_matrix((len(ages), ) * 2)
    M.setdiag(mortalityRV.hazard(ages))

    # Birth
    birthRV = birth.birth_gen(parameters, _findBirthScaling = False)
    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in xrange(len(ages)):
        bj = lambda t: ((1. - parameters.probabilityOfMaleBirth)
                        * birthRV.hazard(t, 0., ages[j] - t))
        result = integrate.quad(bj, 0., 1., limit = 100)
        B_bar[0, j] = result[0]

    return (ages, (B_bar, A, M))


def findDominantEigenpair(parameters,
                          _birthScaling = 1., _matrices = None,
                          *args, **kwargs):
    if _matrices is None:
        _matrices = buildMatrices(parameters, *args, **kwargs)

    (ages, (B_bar, A, M)) = _matrices

    G = _birthScaling * B_bar + A - M

    # Find the largest real eigenvalue.
    [L, V] = sparse.linalg.eigs(G, k = 1, which = 'LR',
                                maxiter = int(1e5))

    l0 = numpy.asscalar(numpy.real_if_close(L[0]))
    v0 = numpy.real_if_close(V[:, 0] / numpy.sum(V[:, 0]))

    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)

    return (l0, (ages, v0))


def findGrowthRate(parameters, *args, **kwargs):
    return findDominantEigenpair(parameters, *args, **kwargs)[0]


@shelved
def findStableAgeStructure(parameters, *args, **kwargs):
    return findDominantEigenpair(parameters, *args, **kwargs)[1]


@shelved
def findBirthScaling(parameters, *args, **kwargs):
    matrices = buildMatrices(parameters, *args, **kwargs)

    def _objective(z):
        return findGrowthRate(parameters,
                              _birthScaling = numpy.asscalar(z),
                              _matrices = matrices,
                              *args, **kwargs)

    scaling = numpy.asscalar(optimize.fsolve(_objective, 0.443))

    return scaling
