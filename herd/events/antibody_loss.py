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


class AntibodyLoss(_event.ForImmuneStatusesMixin,
                   _event.ToImmuneStatusesEvent):
    '''A buffalo losing apparent antibodies.'''

    for_immune_statuses = frozenset({'recovered_true',
                                     'recovered_false'})

    to_immune_statuses = (
        'lost_immunity_false',  # .to_immune_status.rvs() == 0
        'lost_immunity_true')   # .to_immune_status.rvs() == 1

    to_immune_status = ToLostImmunityTrue.name

    waiting_time = WaitingTime.name
