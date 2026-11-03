import scipy

from .. import _event, _rv
from . import recovery_to_carrier


class RV(_rv.TimeSinceEntryDependentWaitingTimeMixin, _rv.ScipyRV):
    '''The waiting time to recovery.'''

    name = 'recovery'

    def rv(self, parameters):
        shape = parameters.recovery_shape
        mean = parameters.recovery_mean
        scale = _rv.gamma_scale_from_mean(mean, shape)
        return scipy.stats.gamma(shape, scale=scale)


class Recovery(_event.ForImmuneStatusMixin, _event.Event):
    '''A buffalo recovering from acute infection.'''

    waiting_time = RV.name

    immune_status = 'infectious'

    def __init__(self, buffalo, at_equilibrium=False):
        self.recovery_to_carrier = getattr(buffalo.herd.rvs,
                                           recovery_to_carrier.RV.name)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        if self.recovery_to_carrier.rvs():
            self.buffalo.change_immune_status_to('carrier')
        else:
            self.buffalo.change_immune_status_to('recovered_true')
