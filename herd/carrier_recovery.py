from scipy.stats import gamma

from .rv import RV


class gen(RV):
    '''Gamma-distributed waiting time to recovery from carrier state with rate
    carrier_recovery_mean and shape carrier_recovery_shape.'''
    def __init__(self, parameters, *args, **kwargs):
        self.carrier_recovery_mean = parameters.carrier_recovery_mean
        self.carrier_recovery_shape = parameters.carrier_recovery_shape
        scale = self.carrier_recovery_mean / self.carrier_recovery_shape
        distn = gamma(self.carrier_recovery_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('carrier_recovery_mean',
                                 'carrier_recovery_shape'))
