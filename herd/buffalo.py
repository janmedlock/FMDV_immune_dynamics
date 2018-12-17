from herd import event


class Events(dict):
    '''Container to hold all events that can happen to a buffalo.'''
    # Consider storing the events in a data type that's more
    # efficient to find the minimum time.
    def __init__(self, buffalo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffalo = buffalo

    def add(self, event_):
        self[event_.name] = event_

    def remove(self, event_):
        del self[event_.name]

    def get_next(self):
        return min(self.values())


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''
    def __init__(self, herd, immune_status='maternal immunity', age=0):
        self.herd = herd
        self.immune_status = immune_status
        self.birth_date = self.herd.time - age
        self.sex = event.Sex(self)
        self.identifier = next(self.herd.identifiers)
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
        self.events = Events(self)
        # Add all valid `_Event()` subclasses.
        # for klass in event._Event.__subclasses__():
        #     try:
        #         self.events.add(klass(self))
        #     except AssertionError:
        #         pass
        self.events.add(event.Mortality(self))
        if self.sex == 'female':
            self.events.add(event.Birth(self))
        if self.immune_status == 'maternal immunity':
            self.events.add(event.MaternalImmunityWaning(self))
        elif self.immune_status == 'susceptible':
            # When building a new herd, the infection time
            # won't be correct because
            # `self.herd.number_infectious` won't be finalized.
            # This gets fixed by `herd.Herd()`
            # calling `update_infection()` after the herd is built.
            self.events.add(event.Infection(self))
        elif self.immune_status == 'exposed':
            self.events.add(event.Progression(self))
        elif self.immune_status == 'infectious':
            self.events.add(event.Recovery(self))
        elif self.immune_status == 'chronic':
            self.events.add(event.ChronicRecovery(self))
        elif self.immune_status == 'recovered':
            self.events.add(event.ImmunityWaning(self))
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

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
