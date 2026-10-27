from scipy.stats import bernoulli

from . import rv


class RV(rv.RV):
    '''Whether a recovery leads to carrier infection is a Bernoulli RV.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.recovery_to_carrier_probability = (
            parameters.recovery_to_carrier_probability)
        distn = bernoulli(self.recovery_to_carrier_probability)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('recovery_to_carrier_probability', ))
