from herd import event
from herd.events import BuffaloEvents


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
        self.herd.remove(self)
        self.events.clear()

    @property
    def age(self):
        return (self.herd.time - self.birth_date)

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_add(self)
