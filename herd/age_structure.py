import numpy
from scipy import stats

from herd.rv import RV
from herd.floquet import find_stable_age_structure


class gen(RV):
    def __init__(self, parameters, *args, **kwargs):
        self._density, self._ages = find_stable_age_structure(parameters,
                                                              *args, **kwargs)
        self._proportion = self._density / self._density.sum()
        self._quantilerv = stats.rv_discrete(
            values = (range(len(self._proportion)), self._proportion),
            *args, **kwargs)

    def rvs(self, *args, **kwargs):
        return self._ages[self._quantilerv.rvs(*args, **kwargs)].squeeze()

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
