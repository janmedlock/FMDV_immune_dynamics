import numpy
from scipy import stats


class RV:
    def _copyattrs(self, obj):
        for x in dir(obj):
            if not hasattr(self, x) and not x.startswith('__'):
                setattr(self, x, getattr(obj, x))

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))

    def __repr__(self, params = ()):
        cls = self.__class__
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramstrs = ['{} = {}'.format(p, getattr(self, p))
                     for p in params]
        if len(params) == 0:
            return '<{}>'.format(clsname)
        else:
            return '<{}: {}>'.format(clsname, ', '.join(paramstrs))


class deterministic(RV, stats.rv_continuous):
    def __init__(self, paramname = '_scale', scale = 1, *args, **kwargs):
        self._scale = scale
        self._paramname = paramname
        stats.rv_continuous.__init__(self, a = scale, b = scale,
                                     *args, **kwargs)

    def _cdf(self, age):
        return numpy.where(age < self._scale, 0, 1)

    def _ppf(self, age):
        return self._scale * numpy.ones_like(age)

    def _rvs(self):
        return self._scale * numpy.ones(self._size)

    def _munp(self, n, *args):
        if n == 1:
            return self._scale
        else:
            return 0

    def __getattribute__(self, k):
        get = object.__getattribute__
        if k == get(self, '_paramname'):
            return get(self, '_scale')
        else:
            return get(self, k)

    def __repr__(self):
        return super().__repr__((self._paramname, ))


class age_structured(RV):
    def __init__(self, ages, proportion, *args, **kwargs):
        self._ages = ages
        self._proportion = proportion
        dages = numpy.hstack((numpy.diff(self._ages),
                              self._ages[-1] - self._ages[-2]))
        self._density = self._proportion / dages
        self._quantilerv = stats.rv_discrete(
            values = (range(len(self._proportion)), self._proportion),
            *args, **kwargs)

    def rvs(self, *args, **kwargs):
        return self._ages[self._quantilerv.rvs(*args, **kwargs)]

    def pdf(self, x):
        return numpy.interp(x, self._ages, self._density,
                            left=0, right=0)

    def logpdf(self, x):
        return numpy.log(self.pdf(x))

    def cdf(self, x):
        x = numpy.asarray(x)
        ix = (self._ages < x[..., numpy.newaxis])
        # The cumulative probability for all ages < x.
        p0 = numpy.where(ix, self._proportion, 0).sum(axis=-1)
        # The cumulative probability from the last age < x
        # up to x.
        a0 = numpy.where(ix, self._ages, 0).max(axis=-1)
        p1 = self.pdf(x) * (x - a0)
        return (p0 + p1)

    def sf(self, x):
        return 1 - self.cdf(x)

    def logsf(self, x):
        return numpy.log(self.sf(x))
