from scipy.stats import gamma

from herd.rv import RV


class gen(RV):
    '''Gamma-distributed progression period with rate
    progression_mean and shape progression_shape.'''
    def __init__(self, parameters, *args, **kwargs):
        self.progression_mean = parameters.progression_mean
        self.progression_shape = parameters.progression_shape
        distn = gamma(self.progression_shape,
                      scale = (self.progression_mean / self.progression_shape))
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('progression_mean', 'progression_shape'))
