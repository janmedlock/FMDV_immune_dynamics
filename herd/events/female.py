import scipy

from .. import _rv


class RV(_rv.ScipyRV):
    '''Whether an offspring is female.'''

    name = 'female'

    def rv(self, parameters):
        p = parameters.female_probability
        return scipy.stats.bernoulli(p)
