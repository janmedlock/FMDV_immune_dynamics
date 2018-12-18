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

    def update(self, iterable):
        super().update(iterable)
        self.herd_events.update(iterable)

    def remove(self, value):
        super().remove(value)
        self.herd_events.remove(value)

    def __del__(self):
        for value in self:
            self.herd_events.remove(value)


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''
    def __init__(self, herd, immune_status='maternal immunity', age=0):
        self.herd = herd
        self.immune_status = immune_status
        self.birth_date = self.herd.time - age
        self.sex = event.Sex(self)
        self.identifier = next(self.herd.identifiers)
        self.events = BuffaloEvents(self)
        if self.herd.debug:
            if age == 0:
                print('t = {}: birth of #{} with status {}'.format(
                    self.herd.time,
                    self.identifier,
                    immune_status))
            else:
                print('t = {}: arrival of #{} at age {} with status {}'.format(
                    self.herd.time,
                    self.identifier,
                    age,
                    immune_status))

    def die(self):
        self.change_immune_status_to('dead')
        self.herd.remove(self)
        del self.events

    @property
    def age(self):
        return (self.herd.time - self.birth_date)

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_add(self)
