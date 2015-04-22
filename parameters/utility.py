import numpy
from scipy import sparse, integrate


def findDominantEigenpair(Y):
    # [L, V] = numpy.linalg.eig(Y)
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
    # A = (- numpy.diag(numpy.hstack((1. / da, 0.)))
    #      + numpy.diag(1. / da, -1))
    # A = sparse.lil_matrix((len(ages), ) * 2)
    # A.setdiag(- numpy.hstack((1. / da, 0.)))
    # A.setdiag(1. / da, -1)

    # Mortality
    # M = numpy.diag(mortality.hazard(ages))
    # M = sparse.lil_matrix((len(ages), ) * 2)
    # M.setdiag(mortality.hazard(ages))

    # Aging & Mortality
    # AM = (- numpy.diag(numpy.hstack((1. / da, 0.)
    #                    + mortality.hazard(ages)))
    #       + numpy.diag(1. / da, -1))
    AM = sparse.lil_matrix((len(ages), ) * 2)
    AM.setdiag(- numpy.hstack((1. / da, 0.))
               - mortality.hazard(ages))
    AM.setdiag(1. / da, -1)

    # B_bar = numpy.zeros((len(ages), ) * 2)
    B_bar = sparse.lil_matrix((len(ages), ) * 2)
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in xrange(len(ages)):
        bj = lambda t: ((1. - male.mean())
                        * birth.hazard(t, 0., ages[j] - t))
        result = integrate.quad(bj, 0., 1., limit = 100)
        B_bar[0, j] = result[0]

    # return (ages, (B_bar, A, M))
    return (ages, (B_bar, AM))


def findGrowthRate(mortality, birth, male,
                   _birthScaling = 1., _matrices = None,
                   *args, **kwargs):
    if _matrices is None:
        (ages, matrices) = buildMatrices(mortality, birth, male,
                                         *args, **kwargs)
    else:
        matrices = _matrices

    # (B_bar, A, M) = matrices
    (B_bar, AM) = matrices

    # G = _birthScaling * B_bar + A - M
    G = _birthScaling * B_bar + AM

    return findDominantEigenpair(G)[0]


def findStableAgeStructure(mortality, birth, male,
                           *args, **kwargs):
    (ages, matrices) = buildMatrices(mortality, birth, male,
                                     *args, **kwargs)

    # (B_bar, A, M) = matrices
    (B_bar, AM) = matrices

    # G = B_bar + A - M
    G = B_bar + AM

    return (ages, findDominantEigenpair(G)[1])


def get_shelve_key(mortality, birth, male):
    # Note: mortality is not yet in the key.
    probabilityOfMaleBirth = male.args[0]
    key = ','.join(str(x) for x in (probabilityOfMaleBirth,
                                    birth.seasonalAmplitude))
    return key


class memoized:
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        args = tuple(a if numpy.isscalar(a) else numpy.asscalar(a)
                     for a in args)
        if args in self.cache:
            return self.cache[args]
        else:
            v = self.func(*args)
            self.cache[args] = v
            return v
