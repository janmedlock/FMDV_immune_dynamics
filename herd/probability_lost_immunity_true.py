from scipy.stats import bernoulli

from .rv import RV


class gen(RV):
    '''Whether losing apparent antibody leads to having true not antibodies
    is a Bernoulli RV.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.probability_lost_immunity_true \
            = parameters.precision_lost_immunity
        distn = bernoulli(self.probability_lost_immunity_true)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('probability_lost_immunity_true', ))
