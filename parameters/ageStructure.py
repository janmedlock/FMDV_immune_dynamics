import numpy
from scipy import stats
import inspect
import os.path
import shelve

from . import utility


class ageStructure_gen(object):
    def __init__(self, mortality, birth, male):
        mydir = os.path.dirname(inspect.getfile(self.__class__))
        shelf = shelve.open(os.path.join(mydir, 'ageStructure.shelve'))

        key = utility.get_shelve_key(mortality, birth, male)

        if key in shelf:
            (self.a, self.p) = shelf[key]
        else:
            (self.a, self.p) = utility.findStableAgeStructure(mortality,
                                                              birth,
                                                              male)
            shelf[key] = (self.a, self.p)

        shelf.close()

        self.rv = stats.rv_discrete(values = (range(len(self.p)), self.p))
        
    def rvs(self, *args, **kwargs):
        return self.a[self.rv.rvs(*args, **kwargs)]
        
    def cdf(self, x):
        return numpy.where(self.a <= x, self.p, 0.).sum()
