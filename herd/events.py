'''Data structures to manage events.'''
from sortedcontainers import SortedSet

from herd import event


class BuffaloEvents(set):
    '''Container to hold all events that can happen to a buffalo.
    Actions on these are copied to `buffalo.herd.events`
    so the next herd event can be found efficiently.'''
    def __init__(self, buffalo):
        super().__init__(event.get_all_valid_events(buffalo))
        self.herd_events = buffalo.herd.events
        self.herd_events.update(self)

    def add(self, value):
        super().add(value)
        self.herd_events.add(value)

    def remove(self, value):
        super().remove(value)
        self.herd_events.remove(value)

    def update(self, *others):
        super().update(*others)
        self.herd_events.update(*others)

    def difference_update(self, *others):
        super().difference_update(*others)
        self.herd_events.difference_update(*others)

    def __del__(self):
        '''When the buffalo dies, remove its events from `herd_events`.'''
        self.herd_events.difference_update(self)
        self.clear()


class HerdEvents(SortedSet):
    '''Container to hold all events that can happen in the herd.
    The `Event()`s are stored sorted by their time so that the
    one with minimum time can be found efficiently.
    `Infection()` events are stored in the `infections` attribute
    since they need to be updated frequently.'''
    def __init__(self):
        super().__init__(key=self.key)
        self.infections = set()

    @staticmethod
    def key(value):
        '''Use `value.time` for the sort key.'''
        return value.time

    @staticmethod
    def is_infection(value):
        '''Test whether `value` is an `event.Infection()`.'''
        return isinstance(value, event.Infection)

    @classmethod
    def get_infections(cls, *iterables):
        '''Filter `event.Infection()` instances from `*iterables`.'''
        return (filter(cls.is_infection, iterable)
                for iterable in iterables)

    def add(self, value):
        super().add(value)
        if self.is_infection(value):
            self.infections.add(value)

    def remove(self, value):
        super().remove(value)
        if self.is_infection(value):
            self.infections.remove(value)

    def update(self, *others):
        super().update(*others)
        self.infections.update(*self.get_infections(*others))

    def difference_update(self, *others):
        super().difference_update(*others)
        self.infections.difference_update(*self.get_infections(*others))

    def get_next(self):
        '''Get the next event, returning `None` if there are no events.'''
        try:
            return self[0]
        except IndexError:
            return None

    def update_infection_times(self):
        '''Update the infection events.'''
        # Use the `super()` versions here to avoid changing `self.infections`.
        # Remove `self.infections` from the `SortedSet`.
        super().difference_update(self.infections)
        # Update the infection times.
        for infection in self.infections:
            infection.time = infection.sample_time()
        # Add the updated `self.infections` to the `SortedSet`.
        super().update(self.infections)
