import scipy

from . import antibody_loss_to_lost_immunity_true, event, rv


class RV(rv.MemorylessWaitingTimeMixin, rv.ScipyRV):
    '''Waiting time to loss of antibodies.'''

    name = 'antibody_loss'

    def rv(self, parameters):
        rate = parameters.antibody_loss_hazard
        scale = rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class AntibodyLoss(event.ForImmuneStatusesMixin, event.Event):
    '''A buffalo losing apparent antibodies.'''

    waiting_time = RV.name

    immune_statuses = frozenset({'recovered_true',
                                 'recovered_false'})

    def __init__(self, buffalo, at_equilibrium=False):
        self.antibody_loss_to_lost_immunity_true = getattr(
            buffalo.herd.rvs,
            antibody_loss_to_lost_immunity_true.RV.name)
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        if self.antibody_loss_to_lost_immunity_true.rvs():
            self.buffalo.change_immune_status_to('lost_immunity_true')
        else:
            self.buffalo.change_immune_status_to('lost_immunity_false')
