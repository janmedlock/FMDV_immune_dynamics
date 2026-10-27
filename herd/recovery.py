import scipy

from . import event, rv


class RV(rv.RV):
    '''Gamma-distributed waiting time to recovery with rate
    recovery_mean and shape recovery_shape.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.recovery_mean = parameters.recovery_mean
        self.recovery_shape = parameters.recovery_shape
        scale = self.recovery_mean / self.recovery_shape
        distn = scipy.stats.gamma(self.recovery_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('recovery_mean', 'recovery_shape'))


class Recovery(event.ForImmuneStatus, event.TimeSinceEntryDependent,
               immune_status='infectious'):
    '''A buffalo recovering from acute infection.'''
    def do(self):
        if self.buffalo.herd.rvs.recovery_to_carrier.rvs():
            self.buffalo.change_immune_status_to('carrier')
        else:
            self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.recovery,
            initializing_herd)
