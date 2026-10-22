from scipy.stats import bernoulli

from .rv import RV


class gen(RV):
    '''Whether an recovery leads to carrier infection is a Bernoulli RV.'''
    def __init__(self, parameters, *args, **kwargs):
        self.probability_carrier = parameters.probability_carrier
        distn = bernoulli(self.probability_carrier)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('probability_carrier', ))
