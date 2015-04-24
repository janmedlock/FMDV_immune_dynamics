import numpy
from scipy import sparse, integrate, optimize
import shelve
import inspect
import os.path
import atexit # Workaround for shelve bug on exit.


class shelved:
    '''
    Memoize results and store to disk.

    This is set up to be used as a decorator:
    @shelved(get_key_func)
    def func(*args, **kwargs):
        ...

    get_key_func() derives the cache key, a string,
    from the arguments to func().
    '''    

    def __init__(self, get_key_func):
        self.get_key_func = get_key_func

    def __del__(self):
        # Catch errors if the cache is already closed.
        try:
            self.cache.close()
            del self.cache
        except (ValueError, TypeError, AttributeError):
            pass

    def __call__(self, func):
        # Put the cache file in the same directory as the caller.
        mydir = os.path.dirname(inspect.getfile(func))
        # Name the cache file func.__name__ + '.shelve'
        myfile = os.path.join(mydir, '{}.shelve'.format(func.__name__))
        self.cache = shelve.open(myfile)
        # Workaround for shelve bug on exit.
        atexit.register(self.cache.close)

        def wrapped_func(*args, **kwargs):
            key = self.get_key_func(*args, **kwargs)

            try:
                return self.cache[key]
            except (KeyError, ValueError, TypeError):
                v = func(*args, **kwargs)
                try:
                    self.cache[key] = v
                except (ValueError, TypeError):
                    # The cache is broken, possibly closed?
                    pass
                return v

        return wrapped_func


def findDominantEigenpair(Y):
    [L, V] = sparse.linalg.eigs(Y, k = 1, which = 'LR',
                                maxiter = int(1e5))

    i = numpy.argmax(numpy.real(L))
    l0 = numpy.asscalar(numpy.real_if_close(L[i]))
    v0 = numpy.real_if_close(V[:, i] / numpy.sum(V[:, i]))

    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)

    return (l0, v0)


def buildMatrices(mortality, birth, male,
                  ageMax = 20., ageStep = 0.01):
    ages = numpy.arange(0., ageMax + ageStep / 2., ageStep)
    da = numpy.diff(ages)

    # Aging
    A = sparse.lil_matrix((len(ages), ) * 2)
    A.setdiag(- numpy.hstack((1. / da, 0.)))
    A.setdiag(1. / da, -1)

    # Mortality
    M = sparse.lil_matrix((len(ages), ) * 2)
    M.setdiag(mortality.hazard(ages))

    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in xrange(len(ages)):
        bj = lambda t: ((1. - male.mean())
                        * birth.hazard(t, 0., ages[j] - t))
        result = integrate.quad(bj, 0., 1., limit = 100)
        B_bar[0, j] = result[0]

    return (ages, (B_bar, A, M))


def findGrowthRate(mortality, birth, male,
                   _birthScaling = 1., _matrices = None,
                   *args, **kwargs):
    if _matrices is None:
        (ages, matrices) = buildMatrices(mortality, birth, male,
                                         *args, **kwargs)
    else:
        matrices = _matrices

    (B_bar, A, M) = matrices

    G = _birthScaling * B_bar + A - M

    return findDominantEigenpair(G)[0]


def get_shelve_key(mortality, birth, male, *args, **kwargs):
    '''
    Derive shelve key from the repr of the mortality, birth,
    and male objects.
    '''
    key = ', '.join(map(repr, (mortality, birth, male)))

    return key


@shelved(get_shelve_key)
def findStableAgeStructure(mortality, birth, male,
                           *args, **kwargs):
    (ages, matrices) = buildMatrices(mortality, birth, male,
                                     *args, **kwargs)

    (B_bar, A, M) = matrices

    G = B_bar + A - M

    evec = findDominantEigenpair(G)[1]

    return (ages, evec)


@shelved(get_shelve_key)
def findBirthScaling(mortality, birth, male,
                     *args, **kwargs):
    # Set birth.scaling to 1. so that this function gives
    # the same answer when run twice.
    birth.scaling = 1.

    (ages, matrices) = buildMatrices(mortality, birth, male,
                                     *args, **kwargs)

    def _objective(z):
        return findGrowthRate(mortality, birth, male,
                              _birthScaling = numpy.asscalar(z),
                              _matrices = matrices,
                              *args, **kwargs)

    scaling = numpy.asscalar(optimize.fsolve(_objective, 0.443))

    return scaling


def fracpart(x):
    return numpy.mod(x, 1.)
