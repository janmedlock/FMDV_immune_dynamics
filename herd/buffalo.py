from scipy import stats

from . import event


class Mortality(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        # Use resampling to get a death age > current age.
        while True:
            age_at_death = self.buffalo.rvs.mortality.rvs()
            if age_at_death > self.buffalo.age():
                break

        self.time = self.buffalo.birth_date + age_at_death

        self.buffalo.events['mortality'] = self

    def __call__(self):
        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].remove(
            self.buffalo)
        self.buffalo.herd.mortality(self.buffalo)


class GiveBirth(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        assert self.buffalo.sex == 'female'

        self.get_next_birth_time()

        self.buffalo.events['give_birth'] = self

    def get_next_birth_time(self):
        self.time = (self.buffalo.herd.time
                      + self.buffalo.rvs.birth.rvs(self.buffalo.herd.time,
                                                   self.buffalo.age()))

    def __call__(self):
        if self.buffalo.immune_status == 'recovered':
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'

        self.buffalo.herd.birth(immune_status = calf_status)

        self.get_next_birth_time()


class MaternalImmunityWaning(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        self.time = (self.buffalo.birth_date
                     + self.buffalo.rvs.maternal_immunity_waning.rvs())

        assert self.time >= self.buffalo.herd.time

        self.buffalo.events['maternal_immunity_waning'] = self

    def __call__(self):
        assert self.buffalo.immune_status == 'maternal immunity'

        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].remove(
            self.buffalo)
        self.buffalo.immune_status = 'susceptible'
        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].append(
            self.buffalo)

        self.buffalo.update_infection_time()

        try:
            del self.buffalo.events['maternal_immunity_waning']
        except KeyError:
            pass


class Recovery(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        self.time = self.buffalo.herd.time + self.buffalo.rvs.recovery.rvs()

        self.buffalo.events['recovery'] = self

    def __call__(self):
        assert self.buffalo.is_infectious()

        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].remove(
            self.buffalo)
        self.buffalo.immune_status = 'recovered'
        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].append(
            self.buffalo)

        try:
            del self.buffalo.events['recovery']
        except KeyError:
            pass


class Infection(event.Event):
    ## Fix me! ##
    def __init__(self, buffalo):
        self.buffalo = buffalo

        assert self.buffalo.is_susceptible()

        if (self.buffalo.herd.number_infectious > 0):
            force_of_infection = (self.buffalo.rvs.transmission_rate
                                  * self.buffalo.herd.number_infectious)

            infection_time = stats.expon.rvs(
                scale = 1 / force_of_infection)

            self.time = self.buffalo.herd.time + infection_time

            self.buffalo.events['infection'] = self
        else:
            try:
                del self.buffalo.events['infection']
            except KeyError:
                pass

    def __call__(self):
        assert self.buffalo.is_susceptible()

        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].remove(
            self.buffalo)
        self.buffalo.immune_status = 'infectious'
        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].append(
            self.buffalo)

        try:
            del self.buffalo.events['infection']
        except KeyError:
            pass
        
        self.buffalo.events['recovery'] = Recovery(self.buffalo)


class Buffalo:
    'A single buffalo and the actions that can occur to it.'

    def __init__(self, herd, immune_status = 'maternal immunity', age = 0,
                 identifier = None, building_herd = False):
        self.herd = herd
        self.immune_status = immune_status

        # All members of the herd have the same parameters.
        self.rvs = self.herd.rvs

        self.birth_date = self.herd.time - age
        self.identifier = identifier

        self.sex = 'male' if (self.rvs.male.rvs() == 1) else 'female'

        self.events = {}

        Mortality(self)

        if self.immune_status == 'maternal immunity':
            MaternalImmunityWaning(self)
        elif self.immune_status == 'susceptible':
            # Defer if building_herd because
            # self.herd.number_infectious won't be correct.
            if not building_herd:
                Infection(self)
        elif self.immune_status == 'infectious':
            Recovery(self)
        elif self.immune_status == 'recovered':
            pass
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

        if self.sex == 'female':
            GiveBirth(self)

        self.herd.by_immune_status[self.immune_status].append(self)

    def age(self):
        return (self.herd.time - self.birth_date)

    def is_susceptible(self):
        return (self.immune_status == 'susceptible')

    def is_infectious(self):
        return (self.immune_status == 'infectious')

    def update_infection_time(self):
        Infection(self)

    def get_next_event(self):
        # Consider storing the events in a data type that's more
        # efficient to find the minimum.
        return  min(self.events.values())
