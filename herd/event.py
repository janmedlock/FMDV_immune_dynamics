'''Events that can happen to a buffalo.'''
from abc import ABC, abstractmethod

import numpy
from scipy import stats

from herd.immune_statuses import immune_statuses


# Events that can happen to each immune status.
# TODO: Build this with metaclass magic.
events_by_immune_status = {immune_status: set()
                           for immune_status in immune_statuses}


class Event(ABC):
    '''Parent class for events that happen to a buffalo.'''
    @abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event.
        Subclasses must define this method.'''

    @abstractmethod
    def do(self):
        '''Execute the event.
        Subclasses must define this method.'''

    @abstractmethod
    def sample_time(self):
        '''Generate a random sample time for the event.
        Subclasses must define this method.'''

    def __init__(self, buffalo):
        self.buffalo = buffalo
        assert self.is_valid()
        self.time = self.sample_time()

    def __call__(self):
        assert self.is_valid()
        self.buffalo.events.remove(self)
        self.do()

    def remove_event(self, cls):
        for e in self.buffalo.events:
            if isinstance(e, cls):
                self.buffalo.events.remove(e)
                break
        else:
            raise RuntimeError(f'Missing {cls.__name__}() event!')

    def __repr__(self):
        return 't = {}: {} for buffalo #{}'.format(
            self.time,
            self.__class__.__name__,
            self.buffalo.identifier)


class Mortality(Event):
    '''A buffalo dying.'''
    def is_valid(self):
        return True  # All buffalo can die.

    def do(self):
        self.buffalo.die()

    def sample_time(self):
        # Use resampling to get a mortality time >= current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.mortality.rvs())
            if time >= self.buffalo.herd.time:
                return time


class Birth(Event):
    '''A buffalo giving birth.'''
    def is_valid(self):
        return self.buffalo.sex == 'female'

    def do(self):
        self.buffalo.give_birth()
        # Add next birth.
        self.time = self.sample_time()
        self.buffalo.events.add(self)

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.birth.rvs(self.buffalo.herd.time,
                                                  self.buffalo.age))


class MaternalImmunityWaning(Event):
    '''A buffalo losing maternal immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'maternal immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')

    def sample_time(self):
        # Use resampling to get a waning time >= current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.maternal_immunity_waning.rvs())
            if time >= self.buffalo.herd.time:
                return time

events_by_immune_status['maternal immunity'].add(MaternalImmunityWaning)


class Infection(Event):
    '''A buffalo becoming infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

    def get_force_of_infection(self):
        return ((self.buffalo.herd.rvs.transmission_rate
                 * self.buffalo.herd.number_infectious)
                + (self.buffalo.herd.rvs.chronic_transmission_rate
                   * self.buffalo.herd.number_chronic))

    def sample_time(self):
        force_of_infection = self.get_force_of_infection()
        # scale = 1 / force_of_infection
        # Handle division by 0.
        if force_of_infection > 0:
            scale = 1 / force_of_infection
        else:
            scale = numpy.inf
        return (self.buffalo.herd.time
                + stats.expon.rvs(scale=scale))

events_by_immune_status['susceptible'].add(Infection)


class Progression(Event):
    '''A buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.progression.rvs())

events_by_immune_status['exposed'].add(Progression)


class Recovery(Event):
    '''A buffalo recovering from acute infection.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'infectious'

    def do(self):
        if self.buffalo.herd.rvs.probability_chronic.rvs() == 1:
            self.buffalo.change_immune_status_to('chronic')
        else:
            self.buffalo.change_immune_status_to('recovered')

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.recovery.rvs())

events_by_immune_status['infectious'].add(Recovery)


class ChronicRecovery(Event):
    '''A buffalo recovering from chronically infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'chronic'

    def do(self):
        self.buffalo.change_immune_status_to('recovered')

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.chronic_recovery.rvs())

events_by_immune_status['chronic'].add(ChronicRecovery)


class AntibodyLoss(Event):
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered'

    def do(self):
        self.buffalo.change_immune_status_to('lost immunity')

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.antibody_loss.rvs())

events_by_immune_status['recovered'].add(AntibodyLoss)


class SecondaryInfection(Infection):
    '''A buffalo becoming infected from lost immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'lost immunity'

    def do(self):
        # Immune status 'lost immunity' can change by either
        # `SecondaryInfection()` or `AntibodyGain()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(AntibodyGain)
        self.buffalo.change_immune_status_to('exposed')

    def get_force_of_infection(self):
        return (self.buffalo.herd.params.lost_immunity_susceptibility
                * super().get_force_of_infection())

events_by_immune_status['lost immunity'].add(SecondaryInfection)


class AntibodyGain(Event):
    def is_valid(self):
        return self.buffalo.immune_status == 'lost immunity'

    def do(self):
        # Immune status 'lost immunity' can change by either
        # `SecondaryInfection()` or `AntibodyGain()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(SecondaryInfection)
        self.buffalo.change_immune_status_to('recovered')

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.antibody_gain.rvs())

events_by_immune_status['lost immunity'].add(AntibodyGain)
