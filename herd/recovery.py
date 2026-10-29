import scipy

from . import event, recovery_to_carrier, rv


class RV(rv.ScipyRV):
    '''The waiting time to recovery.'''

    name = 'recovery'

    def rv(self, parameters):
        shape = parameters.recovery_shape
        mean = parameters.recovery_mean
        scale = rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class Event(event.ForImmuneStatusMixin,
            event.TimeSinceEntryDependentEvent):
    '''A buffalo recovering from acute infection.'''

    waiting_time = RV.name

    immune_status = 'infectious'

    def __init__(self, buffalo, initializing_herd=False):
        self.recovery_to_carrier = getattr(buffalo.herd.rvs,
                                           recovery_to_carrier.RV.name)
        super().__init__(buffalo, initializing_herd=initializing_herd)

    def do(self):
        if self.recovery_to_carrier.rvs():
            self.buffalo.change_immune_status_to('carrier')
        else:
            self.buffalo.change_immune_status_to('recovered_true')
