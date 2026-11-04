import scipy

from .. import _event, _rv


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin,
                  _rv.ScipyRV):
    '''Time to waning of maternal immunity.'''

    name = 'maternal_immunity_waning'

    def rv(self, parameters):
        shape = parameters.maternal_immunity_duration_shape
        mean = parameters.maternal_immunity_duration_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class MaternalImmunityWaning(_event.ForImmuneStatusMixin,
                             _event.ToImmuneStatusEvent):
    '''A buffalo losing maternal immunity.'''

    for_immune_status = 'maternal_immunity'

    to_immune_status = 'susceptible'

    waiting_time = WaitingTime.name
