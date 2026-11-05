'''Base class and helpers for events that can happen to a buffalo.'''

import abc
import functools
import inspect
import itertools
import types

from . import immune_status, sex

# Non-abstract subclasses of `Event()` are added to these sets when
# those events are defined by `ForAllMixin()`, `ForSexMixin()`,
# `ForImmuneStatusMixin()`, and `ForImmuneStatusesMixin()`.
# `Event()`s that can happen to any buffalo.
_events_for_all = set()
# `Event()`s that can happen to each sex.
_events_for_sex = {s: set() for s in sex.sexes}
# `Event()`s that can happen to each immune status.
_events_for_immune_status = {s: set() for s in immune_status.immune_statuses}


def get_events_for(buffalo):
    '''Get all the types of events that can happen to `buffalo`.'''
    return _events_for_all.union(
        _events_for_sex[buffalo.sex],
        _events_for_immune_status[buffalo.immune_status])


def get_events_for_immune_status(immune_status_):
    '''Get the types of events that can happen to `immune_status_`.'''
    return _events_for_immune_status[immune_status_]


def union_of_events_for():
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
        '''The type of the waiting time in
        `.random_variables.RandomVariables()`. Subclasses must define
        this attribute.'''

    @abc.abstractmethod
    def do(self):
        '''Execute the event. Subclasses must define this method.'''

    @abc.abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event. This
        is defined by `ForAllMixin()`, `ForSexMixin()`,
        `ForImmuneStatusMixin()`, and `ForImmuneStatusesMixin()`.'''

    # Arguments to pass to `.waiting_time.rvs()`.
    # Subclasses may ovewrite these, including with descriptors.
    waiting_time_rvs_args = ()
    waiting_time_rvs_kwds = types.MappingProxyType({})

    @classmethod
    def __init_subclass__(cls, abstract=False, **kwds):
        '''Verify that non-abstract `cls` was added to one of the
        `_events_for_*` and that abstract `cls` passed `abstract=True`.'''
        super().__init_subclass__(**kwds)
        if inspect.isabstract(cls):
            assert abstract
        else:
            assert cls in union_of_events_for(), \
                f'`{cls}` not in any `_events_for_*`'

    def __init__(self, buffalo, at_equilibrium=False):
        self.buffalo = buffalo
        # Set the instance attribute `.waiting_time` to the
        # waiting-time random variable from the class attribute
        # `.waiting_time` holding the type of that random variable
        self.waiting_time = self.buffalo.herd.rvs[self.waiting_time]
        assert self.is_valid()
        self.time = self.sample_time(use_time_remaining=at_equilibrium)

    def __call__(self):
        assert self.is_valid()
        self.buffalo.events.remove(self)
        self.do()

    def sample_waiting_time(self, use_time_remaining=False):
        '''Generate a random sample waiting time for the event.'''
        args = self.waiting_time_rvs_args
        kwds = self.waiting_time_rvs_kwds
        if use_time_remaining:
            return self.waiting_time.time_remaining_rvs(self.buffalo.age,
                                                        *args, **kwds)
        return self.waiting_time.rvs(*args, **kwds)

    def sample_time(self, use_time_remaining=False):
        '''Generate a random sample time for the event.'''
        return (
            self.buffalo.herd.time
            + self.sample_waiting_time(use_time_remaining=use_time_remaining))

    def __repr__(self):
        return (f't={self.time:g}: '
                + f'{type(self).__name__} of #{self.buffalo.identifier}')



class ToImmuneStatusEvent(Event,
                          abstract=True):
    '''Base for an `Event()` to transition to an immune status.'''
    @property
    @abc.abstractmethod
    def to_immune_status(self):
        '''The destination immune status. Subclasses must define this
        attribute.'''

    def do(self):
        self.buffalo.change_immune_status_to(self.to_immune_status)


class ToImmuneStatusesEvent(Event,
                            abstract=True):
    '''Base for an `Event()` to transitions to one of multiple immune
    statuses.'''
    @property
    @abc.abstractmethod
    def to_immune_status(self):
        '''The type of the random variable that determines the
        destination immune status in
        `.random_variables.RandomVariables()`. Subclasses must define
        this attribute.'''

    def __init__(self, buffalo, at_equilibrium=False):
        # Set the instance attribute `.to_immune_status` to the random
        # variable from the class attribute `.to_immune_status`
        # holding the type of that random variable
        self.to_immune_status = buffalo.herd.rvs[self.to_immune_status]
        super().__init__(buffalo, at_equilibrium=at_equilibrium)

    def do(self):
        self.buffalo.change_immune_status_to(self.to_immune_status.rvs())


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
    def for_sex(self):
        '''Sex for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to `_events_for_sex[cls.sex]`.'''
        if not inspect.isabstract(cls):
            _events_for_sex[cls.for_sex].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.sex == self.for_sex


class ForImmuneStatusMixin(metaclass=abc.ABCMeta):
    '''Mixin for an event that is valid for one immune status.'''
    @property
    @abc.abstractmethod
    def for_immune_status(self):
        '''Immune status for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to
        `_events_for_immune_status[cls.immune_status]`.'''
        if not inspect.isabstract(cls):
            _events_for_immune_status[cls.for_immune_status].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.immune_status == self.for_immune_status


class ForImmuneStatusesMixin(metaclass=abc.ABCMeta):
    '''Mixin for an event that is valid for some immune statuses.'''
    @property
    @abc.abstractmethod
    def for_immune_statuses(self):
        '''Immune statuses for which this event is valid.'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to
        `_events_for_immune_status[immune_status]`
        for `immune_status` in `.immune_statuses`.'''
        if not inspect.isabstract(cls):
            for immune_status_ in cls.for_immune_statuses:
                _events_for_immune_status[immune_status_].add(cls)
        super().__init_subclass__(**kwds)

    def is_valid(self):
        return self.buffalo.immune_status in self.for_immune_statuses
