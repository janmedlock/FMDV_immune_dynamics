'''Events that can happen to a buffalo.'''

import abc
import functools

import numpy
from scipy import stats

from herd import buffalo


# Use __lt__ and __eq__ to generate all the other comparisons.
@functools.total_ordering
class _Event(abc.ABC):
    '''Parent class for events that happen to a buffalo.'''
    @abc.abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event.
        Subclasses must define this method.'''

    @abc.abstractmethod
    def do(self):
        '''Execute the event.
        Subclasses must define this method.'''

    @abc.abstractmethod
    def sample_time(self):
        '''Generate a random sample time for the event.
        Subclasses must define this method.'''

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self, buffalo_):
        self.buffalo = buffalo_
        assert self.is_valid()
        self.time = self.sample_time()

    def __call__(self):
        assert self.is_valid()
        self.do()

    def __lt__(self, other):
        return (self.time < other.time)

    def __eq__(self, other):
        return (self.time == other.time)

    def __repr__(self):
        return 't = {}: {} for buffalo #{}'.format(
            self.time, self.name, self.buffalo.identifier)


def get_all_valid_events(buffalo_):
    events = []
    # Collect all valid subclasses of `_Event()`.
    for klass in _Event.__subclasses__():
        # `_Event.__init__()` raises an `AssertionError`
        # if `_Event.is_valid()` is False.
        try:
            events.append(klass(buffalo_))
        except AssertionError:
            pass
    return events


def Sex(buffalo_):
    '''A buffalo having its sex determined.'''
    # This is intentionally not an `_Event()`,
    # because it doesn't have a sample time, etc.
    if buffalo_.herd.rvs.female.rvs() == 1:
        return 'female'
    else:
        return 'male'


class Mortality(_Event):
    '''A buffalo dying.'''
    def is_valid(self):
        return True  # All buffalo can die.

    def do(self):
        self.buffalo.herd.remove(self.buffalo)

    def sample_time(self):
        # Use resampling to get a mortality time > current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.mortality.rvs())
            if time > self.buffalo.herd.time:
                return time


class Birth(_Event):
    '''A buffalo giving birth.'''
    # Which classes give mom antibodies that she passes on.
    has_antibodies = ('chronic', 'recovered')

    def is_valid(self):
        return self.buffalo.sex == 'female'

    def do(self):
        if self.buffalo.immune_status in self.has_antibodies:
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'
        self.buffalo.herd.append(buffalo.Buffalo(self.buffalo.herd,
                                                 calf_status))
        # Update to time of the next birth.
        self.time = self.sample_time()

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.birth.rvs(self.buffalo.herd.time,
                                                  self.buffalo.age))


class MaternalImmunityWaning(_Event):
    '''A buffalo losing maternal immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'maternal immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        # Use resampling to get a waning time > current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.maternal_immunity_waning.rvs())
            if time >= self.buffalo.herd.time:
                return time


class Infection(_Event):
    '''A buffalo becoming infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')
        self.buffalo.events.remove(self)
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


class Progression(_Event):
    '''A buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Recovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.progression.rvs())


class Recovery(_Event):
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
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def do_chronic_progression(self):
        self.buffalo.change_immune_status_to('chronic')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ChronicRecovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.recovery.rvs())


class ChronicRecovery(_Event):
    '''A buffalo recovering from chronically infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'chronic'

    def do(self):
        self.buffalo.change_immune_status_to('recovered')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.chronic_recovery.rvs())


class ImmunityWaning(_Event):
    '''A buffalo losing its acquired immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.immunity_waning.rvs())
