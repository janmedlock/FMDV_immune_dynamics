from scipy.stats import bernoulli

from . import rv


class RV(rv.RV):
    '''Whether an offspring is female is a Bernoulli RV.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.female_probability = parameters.female_probability
        distn = bernoulli(self.female_probability)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('female_probability', ))
