from herd import events


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''

    # The immune statuses that give mom antibodies
    # that she passes on to her calves.
    has_antibodies = frozenset({'chronic', 'recovered'})

    def __init__(self, herd, immune_status='maternal immunity', age=0):
        self.herd = herd
        self.immune_status = immune_status
        self.birth_date = self.herd.time - age
        self.sex = 'female' if (self.herd.rvs.female.rvs() == 1) else 'male'
        self.identifier = next(self.herd.identifiers)
        self.events = events.BuffaloEvents(self)
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

    def die(self):
        self.herd.remove(self)
        self.events.clear()

    def give_birth(self):
        assert self.sex == 'female'
        if self.immune_status in self.has_antibodies:
            immune_status_calf = 'maternal immunity'
        else:
            immune_status_calf = 'susceptible'
        self.herd.add(Buffalo(self.herd, immune_status_calf))

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_add(self)
        self.events.add_events_for_immune_status(self, new_immune_status)
