from scipy.stats import bernoulli

from . import rv


class RV(rv.RV):
    '''Whether losing apparent antibody leads to having true not antibodies
    is a Bernoulli RV.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_loss_to_lost_immunity_true_probability = (
            parameters.antibody_loss_to_lost_immunity_true_probability)
        distn = bernoulli(self.antibody_loss_to_lost_immunity_true_probability)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(
            ('antibody_loss_to_lost_immunity_true_probability', ))
