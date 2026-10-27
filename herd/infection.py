import numpy

from . import event


class RV:
    '''Time to infection is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        # The transmission rates are scaled so that R_0 is constant as
        # initial population size varies
        self.carrier_transmission_rate = (parameters.carrier_transmission_rate
                                          / parameters.population_size)
        self.transmission_rate = (parameters.transmission_rate
                                  / parameters.population_size)
        self.rng = numpy.random.default_rng(seed)

    def rvs(self, number_infectious, number_carrier, susceptibility=1,
            size=None):
        force_of_infection = (susceptibility
                * (self.transmission_rate * number_infectious
                   + self.carrier_transmission_rate * number_carrier))
        scale = 1 / force_of_infection if force_of_infection > 0 else numpy.inf
        return self.rng.exponential(scale)


class Infection(event.Memoryless):
    '''Base class for a buffalo becoming infected.'''
    def __init__(self, buffalo, susceptibility=1, initializing_herd=False):
        self.susceptibility = susceptibility
        super().__init__(buffalo, initializing_herd)

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.infection,
            self.buffalo.herd.number_infectious,
            self.buffalo.herd.number_carrier,
            self.susceptibility)


class PrimaryInfection(event.ForImmuneStatus, Infection,
                       immune_status='susceptible'):
    '''A buffalo becoming infected from 'susceptible'.'''
    def __init__(self, buffalo, initializing_herd=False):
        susceptibility = 1
        super().__init__(buffalo, susceptibility, initializing_herd)


class SecondaryInfection(event.ForImmuneStatuses, Infection,
                         immune_statuses={'lost_immunity_true',
                                          'recovered_false'}):
    '''A buffalo becoming infected from 'lost_immunity_true'
    or 'recovered_false'.'''
    def __init__(self, buffalo, initializing_herd=False):
        super().__init__(buffalo,
                         buffalo.herd.params.lost_immunity_susceptibility,
                         initializing_herd)
