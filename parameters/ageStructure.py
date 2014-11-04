import numpy
from scipy import stats
import os.path
import cPickle

from .mortality import *
from .birth import *
from .male import *


class ageStructure_gen(object):
    def __init__(self):
        # (self.a, self.p) = utility.findStableAgeStructure(mortality,
        #                                                   birth,
        #                                                   male)
        (d, f) = os.path.split(__file__)
        (self.a, self.p) = cPickle.load(open(os.path.join(d, 'ageStructure.p')))

        self.rv = stats.rv_discrete(values = (range(len(self.p)), self.p))
        
    def rvs(self, *args, **kwargs):
        return self.a[self.rv.rvs(*args, **kwargs)]
        
    def cdf(self, x):
        return numpy.where(self.a <= x, self.p, 0.).sum()


ageStructure = ageStructure_gen()
