from . import event, events, immune_status


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''

    def __init__(self, herd, immune_status_, age=0, initializing_herd=False):
        self.herd = herd
        self.immune_status = immune_status_
        self.birth_date = self.herd.time - age
        self.sex = 'female' if self.herd.rvs.female.rvs() else 'male'
        self.identifier = next(self.herd.identifiers)
        self.events = events.Buffalo(self)
        self._add_events_on_init(initializing_herd)
        if self.herd.debug:
            print(f't={self.herd.time:g}:'
                  + ' ' + ('birth' if age == 0 else 'arrival')
                  + f' of #{self.identifier}'
                  + (f' at {age=:g}' if age > 0 else '')
                  + f' with immune_status={immune_status_}')

    def _add_events_on_init(self, initializing_herd):
        '''Add an event of each type that can happen to this buffalo.'''
        event_types = event.events_for_all.union(
            event.events_for_sex[self.sex],
            event.events_for_immune_status[self.immune_status])
        for event_type in event_types:
            self.events.add(event_type(self, initializing_herd))

    @property
    def age(self):
        return (self.herd.time - self.birth_date)

    def die(self):
        self.herd.remove(self)
        self.events.clear()

    def give_birth(self):
        assert self.sex == 'female'
        birth_immune_status = (
            immune_status.birth_immune_status[self.immune_status])
        self.herd.add(Buffalo(self.herd, birth_immune_status))

    def _remove_events_on_immune_status_change(self, event_types_old,
                                               event_types_new):
        # Must be a `tuple()` for `isinstance()`
        event_types_to_remove = tuple(
            event_types_old.difference(event_types_new))
        # Cannot `.events.remove()` during iteration of `.events`
        events_to_remove = [event_ for event_ in self.events
                            if isinstance(event_, event_types_to_remove)]
        for event_ in events_to_remove:
            self.events.remove(event_)

    def _add_events_on_immune_status_change(self, event_types_old,
                                            event_types_new):
        event_types_to_add = event_types_new.difference(event_types_old)
        for event_type in event_types_to_add:
            self.events.add(event_type(self))

    def change_immune_status_to(self, immune_status_new):
        immune_status_old = self.immune_status
        (event_types_old, event_types_new) = (
            event.events_for_immune_status[immune_status]
            for immune_status in (immune_status_old, immune_status_new))
        self._remove_events_on_immune_status_change(event_types_old,
                                                    event_types_new)
        self.herd.immune_status_remove(self)
        self.immune_status = immune_status_new
        self.herd.immune_status_add(self)
        self._add_events_on_immune_status_change(event_types_old,
                                                 event_types_new)
