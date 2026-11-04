from .. import _event, _rv


class WaitingTime(_rv.MemorylessWaitingTimeMixin,
                  _rv.RV):
    '''Time to infection is exponential distributed.'''
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
        scale = _rv.expon_scale_from_rate(force_of_infection)
        return self.rng.exponential(scale)


class Infection(_event.ToImmuneStatusEvent,
                abstract=True):
    '''Base class for a buffalo becoming infected.'''
    to_immune_status = 'exposed'
    waiting_time = WaitingTime

    @property
    def waiting_time_rvs_args(self):
        '''Arguments to pass to `.waiting_time.rvs()`.'''
        return (self.buffalo.herd.number_infectious,
                self.buffalo.herd.number_carrier,
                self.susceptibility)

    def __init__(self, buffalo, susceptibility=1, at_equilibrium=False):
        self.susceptibility = susceptibility
        super().__init__(buffalo, at_equilibrium=at_equilibrium)


class PrimaryInfection(_event.ForImmuneStatusMixin,
                       Infection):
    '''A buffalo becoming infected from 'susceptible'.'''
    for_immune_status = 'susceptible'

    def __init__(self, buffalo, at_equilibrium=False):
        super().__init__(buffalo, susceptibility=1,
                         at_equilibrium=at_equilibrium)


class SecondaryInfection(_event.ForImmuneStatusesMixin,
                         Infection):
    '''A buffalo becoming infected from 'lost_immunity_true'
    or 'recovered_false'.'''
    for_immune_statuses = frozenset({'lost_immunity_true',
                                     'recovered_false'})

    def __init__(self, buffalo, at_equilibrium=False):
        super().__init__(buffalo,
                         buffalo.herd.parameters.lost_immunity_susceptibility,
                         at_equilibrium=at_equilibrium)


class HerdMixin:
    '''Mixin for '..herd_.Herd()`.'''
    def update_infection_times(self):
        '''Calculate the number of infectious and carrier buffalo,
        then, if these are different than previous, update
        the infection times for susceptible buffalo.'''
        needs_update = False
        number_infectious_new = len(self.by_immune_status['infectious'])
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            needs_update = True
        number_carrier_new = len(self.by_immune_status['carrier'])
        if number_carrier_new != self.number_carrier:
            self.number_carrier = number_carrier_new
            needs_update = True
        if needs_update:
            # There's a descriptor on `self.events.infections`
            # that handles getting and setting
            infections = self.events.infections
            for infection in infections:
                infection.time = infection.sample_time()
            self.events.infections = infections
