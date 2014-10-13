import numpy


def findDominantEigenvalue(Y):
    [L, V] = numpy.linalg.eig(Y)
    i = numpy.argmax(numpy.real(L))
    l0 = numpy.asscalar(numpy.real_if_close(L[i]))
    assert numpy.isreal(l0), 'Complex dominant eigenvalue!'
    return l0


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
