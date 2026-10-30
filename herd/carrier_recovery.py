import scipy

from . import event, rv


class RV(rv.TimeSinceEntryDependentWaitingTimeMixin, rv.ScipyRV):
    '''The waiting time to recovery from carrier.'''

    name = 'carrier_recovery'

    def rv(self, parameters):
        shape = parameters.carrier_recovery_shape
        mean = parameters.carrier_recovery_mean
        scale = rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class CarrierRecovery(event.ForImmuneStatusMixin, event.Event):
    '''A buffalo recovering from carrier infected.'''

    waiting_time = RV.name

    immune_status = 'carrier'

    def do(self):
        self.buffalo.change_immune_status_to('recovered_true')
