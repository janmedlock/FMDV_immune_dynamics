'''Data structures to manage events.'''

import itertools
import operator

import numpy
import sortedcontainers

from .events import infection


class BuffaloEvents(set):
    '''Container to hold all events that can happen to a buffalo.
    Actions on these are copied to `buffalo.herd.events`
    so the next herd event can be found efficiently.'''
    def __init__(self, buffalo):
        super().__init__()
        self.herd_events = buffalo.herd.events

    def add(self, event):
        super().add(event)
        self.herd_events.add(event)

    def remove(self, event):
        super().remove(event)
        self.herd_events.remove(event)

    def update(self, *others):
        for iterable in others:
            # Make 2 copies in case `iterable` is an iterator.
            (iterable_b, iterable_h) = itertools.tee(iterable, 2)
            super().update(iterable_b)
            self.herd_events.update(iterable_h)

    def clear(self):
        for event in self:
            self.herd_events.remove(event)
        super().clear()


class _SortedEventList(sortedcontainers.SortedKeyList):
    '''Sort items in order by the value of their `.time` attribute.'''
    # `.copy()` passes `key=`
    def __init__(self, *args, key=operator.attrgetter('time'), **kwds):
        super().__init__(*args, key=key, **kwds)


class _Infections:
    '''Descriptor for `Herd.infections`.'''
    attr = '_infections'
    factory = _SortedEventList

    def __get__(self, herd, herd_class=None):
        if herd is None:
            return herd_class
        try:
            return getattr(herd, self.attr)
        except AttributeError:
            pass
        # Set to a new `.factory()` and return that
        store = self.factory()
        setattr(herd, self.attr, store)
        return store

    def __set__(self, herd, infections):
        store = self.__get__(herd)
        if infections is store:
            # `.copy()` to trigger resorting
            setattr(herd, self.attr, infections.copy())
            return
        # In-place replace the values
        store.clear()
        store.update(infections)


class HerdEvents(_SortedEventList):
    '''Container to hold all events that can happen in the herd.'''

    # The `Event()`s are stored sorted by their time so that the one
    # with minimum time can be found efficiently.
    # `infection.Event()`s are stored separately in
    # `.infections` because they need to be updated frequently.
    infections = _Infections()

    def __init__(self, *args, seed=None, **kwds):
        super().__init__(*args, **kwds)
        # To choose randomly if there are multiple events at the minimum time
        self.rng = numpy.random.default_rng(seed)

    @staticmethod
    def __new__(cls, *args, seed=None, **kwds):
        return super().__new__(cls, *args, **kwds)

    @staticmethod
    def _is_infection(event):
        return isinstance(event, infection.Infection)

    def add(self, event):
        if self._is_infection(event):
            self.infections.add(event)
        else:
            super().add(event)

    def remove(self, event):
        if self._is_infection(event):
            self.infections.remove(event)
        else:
            super().remove(event)

    def update(self, *others):
        for iterable in others:
            grouper = itertools.groupby(iterable, self._is_infection)
            for (is_infection, group) in grouper:
                if is_infection:
                    self.infections.update(group)
                else:
                    super().update(group)

    def clear(self):
        super().clear()
        self.infections.clear()

    def next(self):
        '''Return the next event.'''
        # From the first items in each non-empty list, find the one
        # that has minimum time
        lists = (self, self.infections)
        events_first = itertools.chain.from_iterable(
            l.islice(stop=1) for l in lists)
        try:
            event_next = min(events_first, key=self.key)
        except ValueError:  # Both `sels` are empty
            raise StopIteration
        # Check for multiple events at the minimum time
        num_events_next = sum(l.bisect_right(event_next)
                              for l in lists)
        if num_events_next == 1:
            return event_next
        # Randomly choose one of the events at the minimum time
        i = self.rng.choice(num_events_next)
        events_next = itertools.chain.from_iterable(
            l.irange(maximum=event_next) for l in lists)
        (event_next_i,) = itertools.islice(events_next, i, i + 1)
        return event_next_i
