import scipy

from . import rv


class RV(rv.ScipyRV):
    '''Whether a recovery leads to carrier infection.'''

    name = 'recovery_to_carrier'

    def rv(self, parameters):
        p = parameters.recovery_to_carrier_probability
        return scipy.stats.bernoulli(p)
