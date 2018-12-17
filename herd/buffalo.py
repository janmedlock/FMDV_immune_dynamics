from herd import event


class Events(dict):
    '''Container to hold all events that can happen to a buffalo.'''
    # Consider storing the events in a data type that's more
    # efficient to find the minimum time.
    def __init__(self, buffalo):
        super().__init__()
        for event_ in event.get_all_valid_events(buffalo):
            self.add(event_)

    def add(self, event_):
        self[event_.name] = event_

    def remove(self, event_):
        del self[event_.name]

    def get_next(self):
        # `self.values()` is a sequence of `Event()`s.
        # These have comparison methods (<, >, =, etc)
        # defined to compare by event time.
        # Thus, `min()` gives the next one to occur.
        return min(self.values())


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''
    def __init__(self, herd, immune_status='maternal immunity', age=0):
        self.herd = herd
        self.immune_status = immune_status
        self.birth_date = self.herd.time - age
        self.sex = event.Sex(self)
        self.identifier = next(self.herd.identifiers)
        self.events = Events(self)
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

    @property
    def age(self):
        return (self.herd.time - self.birth_date)

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_append(self)

    def update_infection(self):
        infection = self.events['Infection']
        assert infection.is_valid()
        infection.time = infection.sample_time()

    def get_next_event(self):
        return self.events.get_next()
