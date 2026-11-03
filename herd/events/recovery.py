import scipy

from .. import _event, _rv


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin, _rv.ScipyRV):
    '''The waiting time to recovery.'''

    name = 'recovery'

    def rv(self, parameters):
        shape = parameters.recovery_shape
        mean = parameters.recovery_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class ToCarrier(_rv.ScipyRV):
    '''Whether a recovery leads to carrier infection.'''

    name = 'recovery_to_carrier'

    def rv(self, parameters):
        p = parameters.recovery_to_carrier_probability
        return scipy.stats.bernoulli(p)


class Recovery(_event.ForImmuneStatusMixin, _event.Event):
    '''A buffalo recovering from acute infection.'''

    waiting_time = WaitingTime.name

    to_carrier = ToCarrier.name

    immune_status = 'infectious'

    def __init__(self, buffalo, at_equilibrium=False):
        self.to_carrier = getattr(buffalo.herd.rvs, self.to_carrier)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        destination = 'carrier' if self.to_carrier.rvs() else 'recovered_true'
        self.buffalo.change_immune_status_to(destination)
