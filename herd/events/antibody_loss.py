import scipy

from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin, _rv.ScipyRV):
    '''Waiting time to loss of antibodies.'''

    name = 'antibody_loss'

    def rv(self, parameters):
        rate = parameters.antibody_loss_hazard
        scale = _rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class ToLostImmunityTrue(_rv.ScipyRV):
    '''Whether losing apparent antibody leads to having true not antibodies.'''

    name = 'antibody_loss_to_lost_immunity_true'

    def rv(self, parameters):
        p = parameters.antibody_loss_to_lost_immunity_true_probability
        return scipy.stats.bernoulli(p)


class AntibodyLoss(_event.ForImmuneStatusesMixin, _event.Event):
    '''A buffalo losing apparent antibodies.'''

    waiting_time = WaitingTime.name

    to_lost_immunity_true = ToLostImmunityTrue.name

    immune_statuses = frozenset({'recovered_true',
                                 'recovered_false'})

    def __init__(self, buffalo, at_equilibrium=False):
        self.to_lost_immunity_true = getattr(buffalo.herd.rvs,
                                             self.to_lost_immunity_true)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        destination = ('lost_immunity_true' if self.to_lost_immunity_true.rvs()
                       else 'lost_immunity_false')
        self.buffalo.change_immune_status_to(destination)
