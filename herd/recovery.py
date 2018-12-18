from scipy.stats import gamma

from herd.rv import RV


class gen(RV):
    '''Gamma-distributed waiting time to recovery with rate
    recovery_mean and shape recovery_shape.'''
    def __init__(self, parameters, *args, **kwargs):
        self.recovery_mean = parameters.recovery_mean
        self.recovery_shape = parameters.recovery_shape
        scale = self.recovery_mean / self.recovery_shape
        distn = gamma(self.recovery_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('recovery_mean', 'recovery_shape'))
