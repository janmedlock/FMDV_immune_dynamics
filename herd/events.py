'''Data structures to manage events.'''

import itertools
import operator

import sortedcontainers

from . import infection


class Buffalo(set):
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
            iterable_b, iterable_h = itertools.tee(iterable, 2)
            super().update(iterable_b)
            self.herd_events.update(iterable_h)

    def clear(self):
        for event in self:
            self.herd_events.remove(event)
        super().clear()


class _SortedTimeList(sortedcontainers.SortedKeyList):
    '''Sort items in order by the value of their `.time` attribute.'''
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds,
                         key=operator.attrgetter('time'))


class Herd(_SortedTimeList):
    '''Container to hold all events that can happen in the herd.'''
    # The `Event()`s are stored sorted by their time so that the
    # one with minimum time can be found efficiently.
    # `Infection()` events are stored separately in `._infections`
    # because they need to be updated frequently.

    def __init__(self):
        super().__init__()
        self._infections = _SortedTimeList()

    @staticmethod
    def _is_infection(event):
        '''Test whether `event` is an `Infection()`.'''
        return isinstance(event, infection.Infection)

    def add(self, event):
        if self._is_infection(event):
            self._infections.add(event)
        else:
            super().add(event)

    def remove(self, event):
        if self._is_infection(event):
            self._infections.remove(event)
        else:
            super().remove(event)

    def update(self, *others):
        for iterable in others:
            grouper = itertools.groupby(iterable, self._is_infection)
            for (is_infection, group) in grouper:
                if is_infection:
                    self._infections.update(group)
                else:
                    super().update(group)

    def _infections_get(self):
        # `sortedcontainers.SortedKeyList()` requires the items' key
        # values not change while they are stored
        return tuple(self._infections)

    def _infections_set(self, infections):
        self._infections.clear()
        self._infections.update(infections)

    # This descriptor is used by `.herd_.update_infection_times()`
    infections = property(_infections_get, _infections_set,
                          doc='The infection events.')

    def get_next(self):
        '''Get the next event. Raises `IndexError` if there are no events.'''
        try:
            return min(itertools.chain(self[:1], self._infections[:1]),
                       key=self.key)
        except ValueError:
            raise IndexError
