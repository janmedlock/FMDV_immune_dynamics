import scipy

from . import event, rv


class RV(rv.TimeSinceEntryDependentWaitingTimeMixin, rv.ScipyRV):
    '''Waiting time for progression.'''

    name = 'progression'

    def rv(self, parameters):
        shape = parameters.progression_shape
        mean = parameters.progression_mean
        scale = rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class Progression(event.ForImmuneStatusMixin, event.Event):
    '''A buffalo becoming infectious.'''

    waiting_time = RV.name

    immune_status = 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')
