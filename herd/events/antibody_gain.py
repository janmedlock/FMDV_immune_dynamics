import scipy

from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin, _rv.ScipyRV):
    '''Waiting time to gain of antibodies.'''

    name = 'antibody_gain'

    def rv(self, parameters):
        rate = parameters.antibody_gain_hazard
        scale = _rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class ToRecoveredTrue(_rv.ScipyRV):
    '''Whether gaining apparent antibody leads to having true antibodies.'''

    name = 'antibody_gain_to_recovered_true'

    def rv(self, parameters):
        p = parameters.antibody_gain_to_recovered_true_probability
        return scipy.stats.bernoulli(p)


class AntibodyGain(_event.ForImmuneStatusesMixin, _event.Event):
    '''A buffalo gaining apparent antibodies.'''

    waiting_time = WaitingTime.name

    to_recovered_true = ToRecoveredTrue.name

    immune_statuses = frozenset({'lost_immunity_true',
                                 'lost_immunity_false'})

    def __init__(self, buffalo, at_equilibrium=False):
        self.to_recovered_true = getattr(buffalo.herd.rvs,
                                         self.to_recovered_true)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        destination = ('recovered_true' if self.to_recovered_true.rvs()
                       else 'recovered_false')
        self.buffalo.change_immune_status_to(destination)
