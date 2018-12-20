'''Events that can happen to a buffalo.'''
from abc import ABC, abstractmethod

import numpy
from scipy import stats


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

    def __repr__(self):
        return 't = {}: {} for buffalo #{}'.format(
            self.time,
            self.__class__.__name__,
            self.buffalo.identifier)


def get_all_valid_events(buffalo):
    events = set()
    # Collect all valid subclasses of `Event()`.
    # `Event.__init__()` raises an `AssertionError`
    # if `Event.is_valid()` is False.
    for cls in Event.__subclasses__():
        try:
            events.add(cls(buffalo))
        except AssertionError:
            pass
    return events


def Sex(buffalo):
    '''A buffalo having its sex determined.'''
    # This is intentionally not an `Event()`,
    # because it doesn't have a sample time, etc.
    if buffalo.herd.rvs.female.rvs() == 1:
        return 'female'
    else:
        return 'male'


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
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        # Use resampling to get a waning time >= current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.maternal_immunity_waning.rvs())
            if time >= self.buffalo.herd.time:
                return time


class Infection(Event):
    '''A buffalo becoming infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')
        self.buffalo.events.add(Progression(self.buffalo))

    def sample_time(self):
        force_of_infection = (
            (self.buffalo.herd.rvs.transmission_rate
             * self.buffalo.herd.number_infectious)
            + (self.buffalo.herd.rvs.chronic_transmission_rate
               * self.buffalo.herd.number_chronic))
        # scale = 1 / force_of_infection
        # Handle division by 0.
        scale = numpy.ma.filled(
            numpy.ma.divide(1, force_of_infection),
            numpy.inf)
        return (self.buffalo.herd.time
                + stats.expon.rvs(scale=scale))


class Progression(Event):
    '''A buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')
        self.buffalo.events.add(Recovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.progression.rvs())


class Recovery(Event):
    '''A buffalo recovering from acute infection.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'infectious'

    def do(self):
        if self.buffalo.herd.rvs.probability_chronic.rvs() == 1:
            self.do_chronic_progression()
        else:
            self.do_recovery()

    def do_recovery(self):
        self.buffalo.change_immune_status_to('recovered')
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def do_chronic_progression(self):
        self.buffalo.change_immune_status_to('chronic')
        self.buffalo.events.add(ChronicRecovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.recovery.rvs())


class ChronicRecovery(Event):
    '''A buffalo recovering from chronically infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'chronic'

    def do(self):
        self.buffalo.change_immune_status_to('recovered')
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.chronic_recovery.rvs())


class ImmunityWaning(Event):
    '''A buffalo losing its acquired immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.immunity_waning.rvs())
