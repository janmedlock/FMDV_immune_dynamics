from scipy import stats

from . import event


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

        if self.immune_status == 'maternal immunity':
            self.set_maternal_immunity_waning_time()
        elif self.immune_status == 'susceptible':
            # Defer if building_herd because
            # self.herd.number_infectious won't be correct.
            if not building_herd:
                self.update_infection_time()
        elif self.immune_status == 'infectious':
            self.set_recovery_time()
        elif self.immune_status == 'recovered':
            pass
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

        self.set_mortality_time()

        if self.sex == 'female':
            self.update_birth_time()

        self.herd.by_immune_status[self.immune_status].append(self)

    def age(self):
        return self.herd.time - self.birth_date

    def is_susceptible(self):
        return self.immune_status == 'susceptible'

    def is_infectious(self):
        return self.immune_status == 'infectious'

    def mortality(self):
        self.herd.by_immune_status[self.immune_status].remove(self)
        self.herd.mortality(self)

    def set_mortality_time(self):
        # Use resampling to get a death age > current age.
        age_ = self.age()
        while True:
            death_age = self.rvs.mortality.rvs()
            if death_age > age_:
                break

        event_time = self.birth_date + death_age
        self.events['mortality'] = event.Event(
            event_time,
            self.mortality,
            'mortality for #{}'.format(self.identifier))

    def give_birth(self):
        assert self.sex == 'female'

        if self.immune_status == 'recovered':
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'

        self.herd.birth(immune_status = calf_status)

        self.update_birth_time()

    def update_birth_time(self):
        assert self.sex == 'female'

        event_time = (self.herd.time
                      + self.rvs.birth.rvs(self.herd.time, self.age()))
        self.events['give_birth'] = event.Event(
            event_time,
            self.give_birth,
            'give birth for #{}'.format(self.identifier))

    def maternal_immunity_waning(self):
        assert self.immune_status == 'maternal immunity'

        self.herd.by_immune_status[self.immune_status].remove(self)
        self.immune_status = 'susceptible'
        self.herd.by_immune_status[self.immune_status].append(self)

        self.update_infection_time()

        try:
            del self.events['maternal_immunity_waning']
        except KeyError:
            pass

    def set_maternal_immunity_waning_time(self):
        event_time = (self.birth_date
                      + self.rvs.maternal_immunity_waning.rvs())
        assert event_time >= self.herd.time
        self.events['maternal_immunity_waning'] = event.Event(
            event_time,
            self.maternal_immunity_waning,
            'maternal immunity waning for #{}'.format(self.identifier))

    def infection(self):
        assert self.is_susceptible()

        self.herd.by_immune_status[self.immune_status].remove(self)
        self.immune_status = 'infectious'
        self.herd.by_immune_status[self.immune_status].append(self)

        try:
            del self.events['infection']
        except KeyError:
            pass
        
        self.set_recovery_time()
    
    ## Fix me! ##
    def update_infection_time(self):
        assert self.is_susceptible()

        if (self.herd.number_infectious > 0):
            force_of_infection = (self.rvs.transmission_rate
                                  * self.herd.number_infectious)

            infection_time = stats.expon.rvs(
                scale = 1 / force_of_infection)

            event_time = self.herd.time + infection_time

            self.events['infection'] = event.Event(
                event_time,
                self.infection,
                'infection for #{}'.format(self.identifier))
        else:
            try:
                del self.events['infection']
            except KeyError:
                pass

    def recovery(self):
        assert self.is_infectious()

        self.herd.by_immune_status[self.immune_status].remove(self)
        self.immune_status = 'recovered'
        self.herd.by_immune_status[self.immune_status].append(self)

        try:
            del self.events['recovery']
        except KeyError:
            pass

    def set_recovery_time(self):
        event_time = self.herd.time + self.rvs.recovery.rvs()
        self.events['recovery'] = event.Event(
            event_time,
            self.recovery,
            'recovery for #{}'.format(self.identifier))

    def get_next_event(self):
        # Consider storing the events in a data type that's more
        # efficient to find the minimum.
        return min(self.events.values())
