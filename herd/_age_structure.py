import numpy
import scipy

from . import _floquet, _rv


class RV(_rv.RV):
    name = 'age_structure'

    def __init__(self, parameters, seed=None, **kwds):
        super().__init__(parameters, seed=seed)
        (self._density, self._ages) = _floquet.find_stable_age_structure(
            parameters, **kwds)
        self._proportion = self._density / self._density.sum()
        self._age_rv = scipy.stats.rv_discrete(
            values=(self._ages, self._proportion),
            seed=self.rng)

    def rvs(self, size=None):
        return self._age_rv.rvs(size=size)

    def pdf(self, x):
        return numpy.interp(x, self._ages, self._density, left=0, right=0)

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
