import scipy

from . import event, rv


class RV(rv.ScipyRV):
    '''Time to waning of maternal immunity.'''

    name = 'maternal_immunity_waning'

    def rv(self, parameters):
        shape = parameters.maternal_immunity_duration_shape
        mean = parameters.maternal_immunity_duration_mean
        scale = rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class Event(event.ForImmuneStatusMixin,
            event.AgeDependentEvent):
    '''A buffalo losing maternal immunity.'''

    waiting_time = RV.name

    immune_status = 'maternal_immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
