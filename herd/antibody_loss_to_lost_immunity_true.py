import scipy

from . import _rv


class RV(_rv.ScipyRV):
    '''Whether losing apparent antibody leads to having true not antibodies.'''

    name = 'antibody_loss_to_lost_immunity_true'

    def rv(self, parameters):
        p = parameters.antibody_loss_to_lost_immunity_true_probability
        return scipy.stats.bernoulli(p)
