from scipy.stats import bernoulli

from herd.rv import RV


class gen(RV):
    '''Whether an recovery leads to chronic infection is a Bernoulli RV.'''
    def __init__(self, parameters, *args, **kwargs):
        self.probability_chronic = parameters.probability_chronic
        distn = bernoulli(self.probability_chronic)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('probability_chronic', ))
