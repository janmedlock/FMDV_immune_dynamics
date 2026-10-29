import scipy

from . import rv


class RV(rv.ScipyRV):
    '''Whether gaining apparent antibody leads to having true antibodies.'''

    name = 'antibody_gain_to_recovered_true'

    def rv(self, parameters):
        p = parameters.antibody_gain_to_recovered_true_probability
        return scipy.stats.bernoulli(p)
