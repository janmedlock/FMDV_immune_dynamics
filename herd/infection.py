from . import event, rv


class RV(rv.RV):
    '''Time to infection is exponential distributed.'''

    name = 'infection'

    def __init__(self, parameters, seed=None):
        super().__init__(parameters, seed=seed)
        # The transmission rates are scaled so that R_0 is constant as
        # initial population size varies
        self.transmission_rate = (parameters.transmission_rate
                                  / parameters.population_size)
        self.carrier_transmission_rate = (parameters.carrier_transmission_rate
                                          / parameters.population_size)

    def rvs(self, number_infectious, number_carrier, susceptibility=1):
        force_of_infection = (susceptibility
                * (self.transmission_rate * number_infectious
                   + self.carrier_transmission_rate * number_carrier))
        scale = rv.expon_scale_from_rate(force_of_infection)
        return self.rng.exponential(scale)


class Infection(event.MemorylessEvent):
    '''Base class for a buffalo becoming infected.'''

    waiting_time = RV.name

    def __init__(self, buffalo, susceptibility=1, initializing_herd=False):
        self.susceptibility = susceptibility
        super().__init__(buffalo, initializing_herd=initializing_herd)

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.'''
        return self.waiting_time.rvs(self.buffalo.herd.number_infectious,
                                     self.buffalo.herd.number_carrier,
                                     self.susceptibility)


class PrimaryInfection(event.ForImmuneStatusMixin,
                       Infection):
    '''A buffalo becoming infected from 'susceptible'.'''

    immune_status = 'susceptible'

    def __init__(self, buffalo, initializing_herd=False):
        super().__init__(buffalo, susceptibility=1,
                         initializing_herd=initializing_herd)


class SecondaryInfection(event.ForImmuneStatusesMixin,
                         Infection):
    '''A buffalo becoming infected from 'lost_immunity_true'
    or 'recovered_false'.'''

    immune_statuses = frozenset({'lost_immunity_true',
                                 'recovered_false'})

    def __init__(self, buffalo, initializing_herd=False):
        super().__init__(buffalo,
                         buffalo.herd.parameters.lost_immunity_susceptibility,
                         initializing_herd=initializing_herd)
