import scipy

from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin,
                  _rv.ScipyRV):
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


class ToRecoveredEvent(_event.ToImmuneStatusesEvent):
    '''Base for a buffalo gaining apparent antibodies.'''

    to_immune_statuses = ('recovered_false',  # .to_immune_status.rvs() == 0
                          'recovered_true')   # .to_immune_status.rvs() == 1

    to_immune_status = ToRecoveredTrue.name


class AntibodyGain(_event.ForImmuneStatusesMixin,
                   ToRecoveredEvent):
    '''A buffalo gaining apparent antibodies.'''

    for_immune_statuses = frozenset({'lost_immunity_true',
                                     'lost_immunity_false'})

    waiting_time = WaitingTime.name
