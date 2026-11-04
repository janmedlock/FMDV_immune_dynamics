from . import _event, _events_containers, immune_status, sex
from .events import birth, mortality


class Buffalo(birth.BuffaloMixin, mortality.BuffaloMixin):
    '''A single buffalo and the events that can occur to it.'''

    def __init__(self, herd, immune_status_, age=0, at_equilibrium=False):
        self.herd = herd
        self.immune_status = immune_status_
        assert self.immune_status in immune_status.immune_statuses
        assert age >= 0
        self.birth_date = self.herd.time - age
        self.sex = self.sex_rvs()
        assert self.sex in sex.sexes
        self.identifier = next(self.herd.identifiers)
        self.events = _events_containers.BuffaloEvents(self)
        # Add an event of each type that can happen to this buffalo
        for event in _event.get_events_for(self):
            self.events.add(event(self, at_equilibrium=at_equilibrium))
        if self.herd.debug:
            print(f't={self.herd.time:g}: '
                  + ('birth' if age == 0 else 'arrival')
                  + f' of #{self.identifier}'
                  + (f' at {age=:g}' if age > 0 else '')
                  + f' with immune_status={immune_status_}')

    @property
    def age(self):
        return (self.herd.time - self.birth_date)

    def _remove_events_on_immune_status_change(self, event_types_old,
                                               event_types_new):
        # Must be a `tuple()` for `isinstance()`
        event_types_to_remove = tuple(
            event_types_old.difference(event_types_new))
        # Cannot `.events.remove()` during iteration of `.events`
        events_to_remove = [event for event in self.events
                            if isinstance(event, event_types_to_remove)]
        for event in events_to_remove:
            self.events.remove(event)

    def _add_events_on_immune_status_change(self, event_types_old,
                                            event_types_new):
        event_types_to_add = event_types_new.difference(event_types_old)
        for event_type in event_types_to_add:
            self.events.add(event_type(self))

    def change_immune_status_to(self, immune_status_new):
        immune_status_old = self.immune_status
        (event_types_old, event_types_new) = map(
            _event.get_events_for_immune_status,
            (immune_status_old, immune_status_new))
        self._remove_events_on_immune_status_change(event_types_old,
                                                    event_types_new)
        self.herd.immune_status_remove(self)
        self.immune_status = immune_status_new
        self.herd.immune_status_add(self)
        self._add_events_on_immune_status_change(event_types_old,
                                                 event_types_new)
