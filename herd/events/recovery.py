import scipy

from .. import _event, _rv
from . import antibody_gain


class WaitingTime(_rv.TimeSinceEntryDependentWaitingTimeMixin,
                  _rv.ScipyRV):
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


class Recovery(_event.ForImmuneStatusMixin,
               antibody_gain.ToRecoveredEvent):
    '''A buffalo recovering from acute infection.'''

    for_immune_status = 'infectious'

    to_carrier = ToCarrier.name

    waiting_time = WaitingTime.name

    def __init__(self, buffalo, at_equilibrium=False):
        # Set the instance attribute `.to_carrier` to the random
        # variable from the class attribute `.to_carrier` holding the
        # name of that random variable
        self.to_carrier = getattr(buffalo.herd.rvs, self.to_carrier)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        if self.to_carrier.rvs():
            self.buffalo.change_immune_status_to('carrier')
        else:
            super().do()  # To 'recovered_true' or 'recovered_false'
