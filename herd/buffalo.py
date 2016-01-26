from . import mortality
from . import birth
from . import maternal_immunity_waning
from . import recovery
from . import infection


class Buffalo:
    'A single buffalo and the actions that can occur to it.'

    def __init__(self, herd, immune_status = 'maternal immunity', age = 0,
                 building_herd = False):
        self.herd = herd
        self.immune_status = immune_status

        self.birth_date = self.herd.time - age

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

        self.events = {}

        self.events['mortality'] = mortality.Event(self)

        if self.immune_status == 'maternal immunity':
            self.events['maternal immunity waning'] \
                = maternal_immunity_waning.Event(self)
        elif self.immune_status == 'susceptible':
            # Defer if building_herd because
            # self.herd.number_infectious won't be correct.
            if not building_herd:
                e = infection.Event(self)
                if e.time is not None:
                    self.events['infection'] = e
        elif self.immune_status == 'infectious':
            self.events['recovery'] = recovery.Event(self)
        elif self.immune_status == 'recovered':
            pass
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

        self.determine_sex()

        if self.sex == 'female':
            self.events['give birth'] = birth.Event(self)

    def age(self):
        return (self.herd.time - self.birth_date)

    def is_susceptible(self):
        return (self.immune_status == 'susceptible')

    def is_infectious(self):
        return (self.immune_status == 'infectious')

    def determine_sex(self):
        self.sex = 'male' if (self.herd.rvs.male.rvs() == 1) else 'female'

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_append(self)

    def update_infection_time(self):
        e = infection.Event(self)
        if e.time is not None:
            self.events['infection'] = e
        else:
            try:
                del self.events['infection']
            except KeyError:
                pass

    def get_next_event(self):
        # Consider storing the events in a data type that's more
        # efficient to find the minimum.
        return min(self.events.values())
