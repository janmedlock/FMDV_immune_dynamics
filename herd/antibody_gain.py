import scipy

from . import _event, _rv, antibody_gain_to_recovered_true


class RV(_rv.MemorylessWaitingTimeMixin, _rv.ScipyRV):
    '''Waiting time to gain of antibodies.'''

    name = 'antibody_gain'

    def rv(self, parameters):
        rate = parameters.antibody_gain_hazard
        scale = _rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class AntibodyGain(_event.ForImmuneStatusesMixin, _event.Event):
    '''A buffalo gaining apparent antibodies.'''

    waiting_time = RV.name

    immune_statuses = frozenset({'lost_immunity_true',
                                 'lost_immunity_false'})

    def __init__(self, buffalo, at_equilibrium=False):
        self.antibody_gain_to_recovered_true = getattr(
            buffalo.herd.rvs,
            antibody_gain_to_recovered_true.RV.name)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        if self.antibody_gain_to_recovered_true.rvs():
            self.buffalo.change_immune_status_to('recovered_true')
        else:
            self.buffalo.change_immune_status_to('recovered_false')
