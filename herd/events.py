'''Data structures to manage events.'''
from itertools import tee

from sortedcontainers import SortedSet

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
        # 'difference_update()' silently ignores events in `self`
        # that aren't in `self.herd_events`.
        # This is not intended, but it's the only batch removal for
        # `SortedSet()`.
        self.herd_events.difference_update(self)
        super().clear()


class HerdEvents(SortedSet):
    '''Container to hold all events that can happen in the herd.'''
    # The `Event()`s are stored sorted by their time so that the
    # one with minimum time can be found efficiently.
    # `Infection()` events are stored in the `infections` attribute
    # since they need to be updated frequently.
    def __init__(self):
        super().__init__(key=self._get_time)
        self._infections = set()

    @staticmethod
    def _get_time(event_):
        '''The key function used to sort `Event()`s by time.'''
        return event_.time

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

    def difference_update(self, *others):
        for iterable in others:
            # Make 2 copies in case `iterable` is an iterator.
            iterable_s, iterable_i = tee(iterable)
            super().difference_update(iterable_s)
            self._infections.difference_update(filter(self._is_infection,
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
        # Remove `self._infections` from the `SortedSet`.
        super().difference_update(self._infections)
        # Update the infection times.
        for infection in self._infections:
            infection.time = infection.sample_time()
        # Add the updated `self._infections` to the `SortedSet`.
        super().update(self._infections)
