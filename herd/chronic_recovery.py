from scipy.stats import gamma

from herd.rv import RV


class gen(RV):
    '''Gamma-distributed waiting time to recovery from chronic state with rate
    chronic_recovery_mean and shape chronic_recovery_shape.'''
    def __init__(self, parameters, *args, **kwargs):
        self.chronic_recovery_mean = parameters.chronic_recovery_mean
        self.chronic_recovery_shape = parameters.chronic_recovery_shape
        scale = self.chronic_recovery_mean / self.chronic_recovery_shape
        distn = gamma(self.chronic_recovery_shape, scale=scale)

    def __repr__(self):
        return super().__repr__(('chronic_recovery_mean',
                                 'chronic_recovery_shape'))
