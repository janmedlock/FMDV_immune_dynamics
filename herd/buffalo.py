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

        self.identifier = self.herd.identifier
        self.herd.identifier += 1

        self.sex = 'male' if (self.herd.rvs.male.rvs() == 1) else 'female'

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

        if self.sex == 'female':
            self.events['give birth'] = birth.Event(self)

        self.herd.by_immune_status[self.immune_status].append(self)

    def age(self):
        return (self.herd.time - self.birth_date)

    def is_susceptible(self):
        return (self.immune_status == 'susceptible')

    def is_infectious(self):
        return (self.immune_status == 'infectious')

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
        return  min(self.events.values())
