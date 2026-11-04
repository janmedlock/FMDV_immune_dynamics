import scipy

from .. import _event, _rv
from . import antibody_gain


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin,
                  _rv.ScipyRV):
    '''The waiting time to recovery from carrier.'''
    def rv(self, parameters):
        shape = parameters.carrier_recovery_shape
        mean = parameters.carrier_recovery_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class CarrierRecovery(_event.ForImmuneStatusMixin,
                      antibody_gain.ToRecoveredEvent):
    '''A buffalo recovering from carrier infected.'''
    for_immune_status = 'carrier'
    waiting_time = WaitingTime
