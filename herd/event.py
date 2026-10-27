'''Base class and helpers for events that can happen to a buffalo.'''

import abc
import inspect
import warnings

from .immune_status import immune_statuses
from .sex import sexes

# `Event()`s are added to these sets when those events are defined,
# throught the `.__init_subclass__()` of the mixins `For*()`
events_for_all = set()
# `Event()`s that can happen to each sex.
events_for_sex = {sex: set() for sex in sexes}
# `Event()`s that can happen to each immune status.
events_for_immune_status = {immune_status: set()
                            for immune_status in immune_statuses}


class Event(metaclass=abc.ABCMeta):
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
    def sample_waiting_time_from(self, rv, *args,
                                 initializing_herd=False, **kwds):
        '''Helper for sample_waiting_time.
        Subclasses must define this method'''

    @abc.abstractmethod
    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.
        Subclasses must define this method.'''

    def __init__(self, buffalo, initializing_herd=False):
        self.buffalo = buffalo
        assert self.is_valid()
        self.time = self.sample_time(initializing_herd)

    def __call__(self):
        assert self.is_valid()
        self.buffalo.events.remove(self)
        self.do()

    def sample_time(self, initializing_herd=False):
        '''Generate a random sample time for the event.'''
        return (
            self.buffalo.herd.time
            + self.sample_waiting_time(initializing_herd))

    def __repr__(self):
        return (
            f't={self.time:g}: '
            + self.__class__.__name__
            + f' of #{self.buffalo.identifier}'
        )


class Memoryless(Event):
    '''An `Event()` with memoryless waiting times.'''
    def sample_waiting_time_from(self, rv, *args,
                                 initializing_herd=False, **kwds):
        '''Helper for sample_waiting_time.'''
        assert not initializing_herd
        return rv.rvs(*args, **kwds)


class AgeDependent(Event):
    '''An `Event()` with waiting times dependent on only age.'''
    def sample_age(self, rv, max_samples=10_000):
        '''Get a sample age that is at least the buffalo's current age.'''
        for _ in range(max_samples):
            age = rv.rvs()
            if age >= self.buffalo.age:
                return age
        raise ValueError(f'Resampling failed after {max_samples=}!')

    def sample_waiting_time_from(self, rv, initializing_herd=False):
        if initializing_herd:
            return self.sample_age(rv) - self.buffalo.age
        return rv.rvs()


class TimeSinceEntryDependent(Event):
    '''An `Event()` with waiting times dependent on only time since entry.'''
    def sample_waiting_time_from(self, rv, initializing_herd=False):
        if initializing_herd:
            # TODO: Fix
            msg = ('`TimeSinceEntryDependent.sample_waiting_time_from'
                   + '(..., initializing_herd=True)` not implemented.')
            warnings.warn(msg)
        return rv.rvs()


class ForAll:
    '''Mixin for an event that is valid for all buffalo.'''
    @classmethod
    def __init_subclass__(cls, **kwds):
        super().__init_subclass__(**kwds)
        if not inspect.isabstract(cls):
            events_for_all.add(cls)

    def is_valid(self):
        return True


class ForSex:
    '''Mixin for an event that is valid for one sex.'''
    @classmethod
    def __init_subclass__(cls, sex, **kwds):
        super().__init_subclass__(**kwds)
        cls.sex = sex
        if not inspect.isabstract(cls):
            events_for_sex[cls.sex].add(cls)

    def is_valid(self):
        return self.buffalo.sex == self.sex


class ForImmuneStatus:
    '''Mixin for an event that is valid for one immune status.'''
    @classmethod
    def __init_subclass__(cls, immune_status, **kwds):
        super().__init_subclass__(**kwds)
        cls.immune_status = immune_status
        if not inspect.isabstract(cls):
            events_for_immune_status[cls.immune_status].add(cls)

    def is_valid(self):
        return self.buffalo.immune_status == self.immune_status


class ForImmuneStatuses:
    '''Mixin for an event that is valid for some immune statuses.'''
    @classmethod
    def __init_subclass__(cls, immune_statuses, **kwds):
        super().__init_subclass__(**kwds)
        cls.immune_statuses = frozenset(immune_statuses)
        if not inspect.isabstract(cls):
            for immune_status in cls.immune_statuses:
                events_for_immune_status[immune_status].add(cls)

    def is_valid(self):
        return self.buffalo.immune_status in self.immune_statuses


