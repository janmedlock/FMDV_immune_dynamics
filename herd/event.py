'''Base class and helpers for events that can happen to a buffalo.'''

import abc
import functools
import inspect
import itertools

from .immune_status import immune_statuses
from .sex import sexes

# Subclasses of `Event()` are added to these sets when those events are
# defined by `ForAllMixin()`, `ForSexMixin()`,
# `ForImmuneStatusMixin()`, and `ForImmuneStatusesMixin()`.
# `Event()`s that can happen to any buffalo.
_events_for_all = set()
# `Event()`s that can happen to each sex.
_events_for_sex = {sex: set() for sex in sexes}
# `Event()`s that can happen to each immune status.
_events_for_immune_status = {immune_status: set()
                             for immune_status in immune_statuses}


def get_events_for(buffalo):
    '''Get all the types of events that can happen to `buffalo`.'''
    return _events_for_all.union(
        _events_for_sex[buffalo.sex],
        _events_for_immune_status[buffalo.immune_status])


def get_events_for_immune_status(immune_status):
    '''Get the types of events that can happen to `immune_status`.'''
    return _events_for_immune_status[immune_status]


def _union_of_events_for():
    '''Get the union of all the events in `_events_for_*`.'''
    return functools.reduce(set.union,
                            itertools.chain.from_iterable(
                                d.values()
                                for d in (_events_for_sex,
                                          _events_for_immune_status)),
                            initial=_events_for_all)


class Event(metaclass=abc.ABCMeta):
    '''Parent class for events that happen to a buffalo.'''
    @property
    @abc.abstractmethod
    def waiting_time(self):
        '''The name of the waiting time in
        `.rv.RandomVariables()`. Subclasses must define this
        attribute.'''

    @abc.abstractmethod
    def do(self):
        '''Execute the event. Subclasses must define this method.'''

    @abc.abstractmethod
    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.  This
        is defined by the subclasses `Memoryless()`, `AgeDependent()`,
        and `TimeSinceEntryDependent()`.'''

    @abc.abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event. This
        is defined by `ForAllMixin()`, `ForSexMixin()`,
        `ForImmuneStatusMixin()`, and `ForImmuneStatusesMixin()`.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Verify that non-abstract `cls` was added to one of the
        `_events_for_*`.'''
        super().__init_subclass__(**kwds)
        if not inspect.isabstract(cls):
            assert cls in _union_of_events_for(), \
                f'`{cls}` not in any `_events_for_*`'

    def __init__(self, buffalo, initializing_herd=False):
        self.buffalo = buffalo
        # Set the instance attribute `.waiting_time`
        # to the waiting-time random variable
        # from the class attribute `.waiting_time`
        # holding the name of that random variable
        self.waiting_time = getattr(self.buffalo.herd.rvs,
                                    self.waiting_time)
        assert self.is_valid()
        self.time = self.sample_time(initializing_herd=initializing_herd)

    def __call__(self):
        assert self.is_valid()
        self.buffalo.events.remove(self)
        self.do()

    def sample_time(self, initializing_herd=False):
        '''Generate a random sample time for the event.'''
        return (
            self.buffalo.herd.time
            + self.sample_waiting_time(initializing_herd=initializing_herd))

    def __repr__(self):
        return (f't={self.time:g}: '
                + f'{type(self).__name__} of #{self.buffalo.identifier}')


class MemorylessEvent(Event):
    '''An `Event()` with memoryless waiting times.'''

    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.'''
        return self.waiting_time.rvs()


class AgeDependentEvent(Event):
    '''An `Event()` with waiting times dependent on only age.'''
    def sample_waiting_time_remaining(self, max_samples=10_000):
        '''Get a sample waiting time remaining given the buffalo has
        survived to its current age.'''
        for _ in range(max_samples):
            time_remaining = self.waiting_time.rvs() - self.buffalo.age
            if time_remaining >= 0:
                return time_remaining
        raise ValueError(f'Resampling failed after {max_samples=}!')

    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.'''
        if initializing_herd:
            return self.sample_waiting_time_remaining()
        return self.waiting_time.rvs()


class TimeSinceEntryDependentEvent(Event):
    '''An `Event()` with waiting times dependent on only time since entry.'''

    def sample_waiting_time_remaining(self, max_samples=10_000):
        '''Get a sample waiting time minus time since entry
        given that the event has not yet occurred.'''
        for _ in range(max_samples):
            time_remaining = (self.waiting_time.rvs()
                              - self.waiting_time.time_since_entry.rvs())
            if time_remaining >= 0:
                return time_remaining
        raise ValueError(f'Resampling failed after {max_samples=}!')

    def sample_waiting_time(self, initializing_herd=False):
        '''Generate a random sample waiting time for the event.'''
        if initializing_herd:
            return self.sample_waiting_time_remaining()
        return self.waiting_time.rvs()


class ForAllMixin:
    '''Mixin for an event that is valid for all buffalo.'''
    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to `_events_for_all`.'''
        if not inspect.isabstract(cls):
            _events_for_all.add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return True


class ForSexMixin(metaclass=abc.ABCMeta):
    '''Mixin for an event that is valid for one sex.'''
    @property
    @abc.abstractmethod
    def sex(self):
        '''Sex for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to `_events_for_sex[cls.sex]`.'''
        if not inspect.isabstract(cls):
            _events_for_sex[cls.sex].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.sex == self.sex


class ForImmuneStatusMixin(metaclass=abc.ABCMeta):
    '''Mixin for an event that is valid for one immune status.'''
    @property
    @abc.abstractmethod
    def immune_status(self):
        '''Immune status for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to
        `_events_for_immune_status[cls.immune_status]`.'''
        if not inspect.isabstract(cls):
            _events_for_immune_status[cls.immune_status].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.immune_status == self.immune_status


class ForImmuneStatusesMixin(metaclass=abc.ABCMeta):
    '''Mixin for an event that is valid for some immune statuses.'''
    @property
    @abc.abstractmethod
    def immune_statuses(self):
        '''Immune statuses for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to
        `_events_for_immune_status[immune_status]`
        for `immune_status` in `cls.immune_statuses`.'''
        if not inspect.isabstract(cls):
            for immune_status in cls.immune_statuses:
                _events_for_immune_status[immune_status].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.immune_status in self.immune_statuses
