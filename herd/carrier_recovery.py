import scipy

from . import event, rv


class RV(rv.RV):
    '''Gamma-distributed waiting time to recovery from carrier state with rate
    carrier_recovery_mean and shape carrier_recovery_shape.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.carrier_recovery_mean = parameters.carrier_recovery_mean
        self.carrier_recovery_shape = parameters.carrier_recovery_shape
        scale = self.carrier_recovery_mean / self.carrier_recovery_shape
        distn = scipy.stats.gamma(self.carrier_recovery_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('carrier_recovery_mean',
                                 'carrier_recovery_shape'))


class CarrierRecovery(event.ForImmuneStatus, event.TimeSinceEntryDependent,
                      immune_status='carrier'):
    '''A buffalo recovering from carrier infected.'''
    def do(self):
        self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.carrier_recovery,
            initializing_herd)
