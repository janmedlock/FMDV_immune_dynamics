from scipy.stats import gamma

from .rv import RV


class gen(RV):
    '''Gamma-distributed progression period with rate
    progression_mean and shape progression_shape.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.progression_mean = parameters.progression_mean
        self.progression_shape = parameters.progression_shape
        scale = self.progression_mean / self.progression_shape
        distn = gamma(self.progression_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('progression_mean', 'progression_shape'))
