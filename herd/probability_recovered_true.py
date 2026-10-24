from scipy.stats import bernoulli

from .rv import RV


class gen(RV):
    '''Whether gaining apparent antibody leads to having true antibodies
    is a Bernoulli RV.'''
    def __init__(self, parameters, *args, **kwargs):
        self.probability_recovered_true = parameters.precision_recovered
        distn = bernoulli(self.probability_recovered_true)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('probability_recovered_true', ))
