import scipy

from .. import _event, _rv
from . import antibody_gain


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin,
                  _rv.ScipyRV):
    '''The waiting time to recovery.'''
    def rv(self, parameters):
        shape = parameters.recovery_shape
        mean = parameters.recovery_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class RecoveryTo(antibody_gain.AntibodyGainTo):
    '''Whether recovery leads to carrier infection or one of the
    recovered states.'''
    def choices(self, parameters):
        p_carrier = parameters.recovery_to_carrier_probability
        choices_recovered = super().choices(parameters)
        return {
            'carrier': p_carrier,
        } | {
            recovered: (1 - p_carrier) * p_recovered
            for (recovered, p_recovered) in choices_recovered.items()
        }


class Recovery(_event.ForImmuneStatusMixin,
               _event.ToImmuneStatusesEvent):
    '''A buffalo recovering from acute infection.'''
    for_immune_status = 'infectious'
    to_immune_status = RecoveryTo
    waiting_time = WaitingTime
