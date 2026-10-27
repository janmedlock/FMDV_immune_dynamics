from scipy.stats import bernoulli

from . import rv


class RV(rv.RV):
    '''Whether gaining apparent antibody leads to having true antibodies
    is a Bernoulli RV.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_gain_to_recovered_true_probability = (
            parameters.antibody_gain_to_recovered_true_probability)
        distn = bernoulli(self.antibody_gain_to_recovered_true_probability)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(
            ('antibody_gain_to_recovered_true_probability', ))
