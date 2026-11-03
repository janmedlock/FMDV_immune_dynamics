import scipy

from .. import _event, _rv


class RV(_rv.TimeSinceEntryDependentWaitingTimeMixin, _rv.ScipyRV):
    '''The waiting time to recovery from carrier.'''

    name = 'carrier_recovery'

    def rv(self, parameters):
        shape = parameters.carrier_recovery_shape
        mean = parameters.carrier_recovery_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class CarrierRecovery(_event.ForImmuneStatusMixin, _event.Event):
    '''A buffalo recovering from carrier infected.'''

    waiting_time = RV.name

    immune_status = 'carrier'

    def do(self):
        self.buffalo.change_immune_status_to('recovered_true')
