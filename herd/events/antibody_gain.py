import scipy

from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin,
                  _rv.ScipyRV):
    '''Waiting time to gain of antibodies.'''
    def rv(self, parameters):
        rate = parameters.antibody_gain_hazard
        scale = _rv.expon_scale_from_rate(rate)
        return scipy.stats.expon(scale=scale)


class AntibodyGainTo(_rv.ChoicesRV):
    '''Whether gaining apparent antibody leads to having true antibodies.'''
    def choices(self, parameters):
        p_true = parameters.recovered_true_probability
        return {'recovered_true' : p_true,
                'recovered_false': 1 - p_true}


class ToRecoveredEvent(_event.ToImmuneStatusesEvent,
                       abstract=True):
    '''Base for a buffalo gaining apparent antibodies.'''
    to_immune_status = AntibodyGainTo


class AntibodyGain(_event.ForImmuneStatusesMixin,
                   ToRecoveredEvent):
    '''A buffalo gaining apparent antibodies.'''
    for_immune_statuses = frozenset({'lost_immunity_true',
                                     'lost_immunity_false'})
    waiting_time = WaitingTime
