'''Data structures to manage events.'''
from itertools import tee
from operator import attrgetter

from sortedcontainers import SortedKeyList

from herd import event


class BuffaloEvents(set):
    '''Container to hold all events that can happen to a buffalo.
    Actions on these are copied to `buffalo.herd.events`
    so the next herd event can be found efficiently.'''
    def __init__(self, buffalo):
        super().__init__()
        self.herd_events = buffalo.herd.events
        # Add all the events that can happen to this buffalo.
        self.update(event.get_all_valid_events(buffalo))

    def add(self, event_):
        super().add(event_)
        self.herd_events.add(event_)

    def remove(self, event_):
        super().remove(event_)
        self.herd_events.remove(event_)

    def update(self, *others):
        for iterable in others:
            # Make 2 copies in case `iterable` is an iterator.
            iterable_b, iterable_h = tee(iterable, 2)
            super().update(iterable_b)
            self.herd_events.update(iterable_h)

    def clear(self):
        for event_ in self:
            self.herd_events.remove(event_)
        super().clear()


class HerdEvents(SortedKeyList):
    '''Container to hold all events that can happen in the herd.'''
    # The `Event()`s are stored sorted by their time so that the
    # one with minimum time can be found efficiently.
    # `Infection()` events are stored in the `infections` attribute
    # since they need to be updated frequently.

    # The key function used to sort `Event()`s by their `time` attribute.
    _get_time = attrgetter('time')

    def __init__(self):
        super().__init__(key=self._get_time)
        self._infections = set()

    @staticmethod
    def _is_infection(event_):
        '''Test whether `event_` is an `event.Infection()`.'''
        return isinstance(event_, event.Infection)

    def add(self, event_):
        super().add(event_)
        if self._is_infection(event_):
            self._infections.add(event_)

    def remove(self, event_):
        super().remove(event_)
        if self._is_infection(event_):
            self._infections.remove(event_)

    def update(self, *others):
        for iterable in others:
            # Make 2 copies in case `iterable` is an iterator.
            iterable_s, iterable_i = tee(iterable)
            super().update(iterable_s)
            self._infections.update(filter(self._is_infection,
                                           iterable_i))

    def get_next(self):
        '''Get the next event.  Returns `None` if there are no events.'''
        try:
            return self[0]
        except IndexError:
            return None

    def update_infection_times(self):
        '''Update the infection events.'''
        # Use the `super()` versions here to avoid changing `self._infections`.
        for infection in self._infections:
            super().remove(infection)
            infection.time = infection.sample_time()
        super().update(self._infections)
