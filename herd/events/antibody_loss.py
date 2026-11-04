import scipy

from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin, _rv.ScipyRV):
    '''Waiting time to loss of antibodies.'''
    def rv(self, parameters):
        rate = parameters.antibody_loss_hazard
        scale = _rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class AntibodyLossTo(_rv.ChoicesRV):
    '''Whether losing apparent antibody leads to having true not antibodies.'''
    def choices(self, parameters):
        p_true = parameters.antibody_loss_to_lost_immunity_true_probability
        return {'lost_immunity_true': p_true,
                'lost_immunity_false': 1 - p_true}


class AntibodyLoss(_event.ForImmuneStatusesMixin,
                   _event.ToImmuneStatusesEvent):
    '''A buffalo losing apparent antibodies.'''
    for_immune_statuses = frozenset({'recovered_true',
                                     'recovered_false'})
    to_immune_status = AntibodyLossTo
    waiting_time = WaitingTime
