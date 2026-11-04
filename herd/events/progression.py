import scipy

from .. import _event, _rv


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin,
                  _rv.ScipyRV):
    '''Waiting time for progression.'''

    name = 'progression'

    def rv(self, parameters):
        shape = parameters.progression_shape
        mean = parameters.progression_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class Progression(_event.ForImmuneStatusMixin,
                  _event.ToImmuneStatusEvent):
    '''A buffalo becoming infectious.'''

    for_immune_status = 'exposed'

    to_immune_status = 'infectious'

    waiting_time = WaitingTime.name
