import numpy
from scipy import stats


class deterministic_gen(stats.rv_continuous):
    def _cdf(self, age):
        return numpy.where(age < 1., 0., 1.)

    def _ppf(self, age):
        return numpy.ones_like(age)
    
    def _rvs(self):
        return numpy.ones(self._size)


deterministic = deterministic_gen(a = 1., b = 1.)
