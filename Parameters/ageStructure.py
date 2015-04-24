import numpy
from scipy import stats

from . import rv
from . import utility


class ageStructure_gen(rv.RV):
    def __init__(self, mortality, birth, male,
                 *args, **kwargs):
        self.findStableAgeStructure(mortality, birth, male,
                                    *args, **kwargs)

        self._quantilerv = stats.rv_discrete(values = (range(len(self.p)),
                                                       self.p))
        
    def rvs(self, *args, **kwargs):
        return self.a[self._quantilerv.rvs(*args, **kwargs)]
        
    def cdf(self, x):
        return numpy.where(self.a <= x, self.p, 0.).sum()

    def findStableAgeStructure(self, mortality, birth, male,
                               *args, **kwargs):
        (self.a, self.p) = utility.findStableAgeStructure(mortality,
                                                          birth,
                                                          male,
                                                          *args,
                                                          **kwargs)
