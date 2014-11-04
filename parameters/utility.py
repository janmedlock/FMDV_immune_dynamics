import numpy
from scipy import sparse, integrate


def findDominantEigenpair(Y):
    # numpy
    [L, V] = numpy.linalg.eig(Y)

    # scipy.sparse
    # [L, V] = sparse.linalg.eigs(Y, k = 1, which = 'LR', maxiter = 20000)

    i = numpy.argmax(numpy.real(L))
    l0 = numpy.asscalar(numpy.real_if_close(L[i]))
    v0 = numpy.real_if_close(V[:, i] / numpy.sum(V[:, i]))

    assert numpy.isreal(l0), 'Complex dominant eigenvalue: {}'.format(l0)

    return (l0, v0)


def buildMatrices(mortality, birth, male,
                  ageMax = 20., ageStep = 0.01):
    ages = numpy.arange(0., ageMax + ageStep / 2., ageStep)

    da = numpy.diff(ages)
    A = - numpy.diag(numpy.hstack((1. / da, 0.))) + numpy.diag(1. / da, -1)

    M = numpy.diag(mortality.hazard(ages))

    B_bar = numpy.zeros((len(ages), len(ages)))
    # The first row, B_bar[0], is the mean, over a year,
    # of the birth rates times the probability of female birth.
    for j in xrange(len(ages)):
        bj = lambda t: (1. - male.mean()) * birth.hazard(t, 0., ages[j] - t)
        result = integrate.quad(bj, 0., 1., limit = 100)
        B_bar[0, j] = result[0]

    return (ages, (B_bar, A, M))


def findGrowthRate(mortality, birth, male,
                   birthScaling = 1., _matrices = None,
                   *args, **kwargs):
    if _matrices is None:
        (ages, (B_bar, A, M)) = buildMatrices(mortality, birth, male,
                                              *args, **kwargs)
    else:
        (B_bar, A, M) = _matrices

    G = birthScaling * B_bar + A - M
    return findDominantEigenpair(G)[0]


def findStableAgeStructure(mortality, birth, male,
                           *args, **kwargs):
    (ages, (B_bar, A, M)) = buildMatrices(mortality, birth, male,
                                          *args, **kwargs)

    G = B_bar + A - M
    return (ages, findDominantEigenpair(G)[1])


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
