from scipy import stats

from . import event


class Buffalo:
    'A single buffalo and the actions that can occur to it.'

    def __init__(self, herd, immune_status = 'maternal immunity', age = 0.,
                 identifier = None):
        self.herd = herd
        self.immune_status = immune_status

        # All members of the herd have the same parameters.
        self.rvs = self.herd.rvs

        self.birth_date = self.herd.time - age
        self.identifier = identifier
        self.sex = 'male' if (self.rvs.male.rvs() == 1) \
          else 'female'

        self.events = {}

        if self.immune_status == 'maternal immunity':
            event_time = (self.birth_date
                          + self.rvs.maternal_immunity_waning.rvs())
            assert event_time >= 0.
            self.events['maternal_immunity_waning'] = event.Event(
                    event_time,
                    self.maternal_immunity_waning,
                    'maternal immunity waning for #{}'.format(self.identifier))
        elif self.immune_status == 'susceptible':
            pass
        elif self.immune_status == 'infectious':
            self.events['recovery'] \
                = event.Event(self.herd.time + self.rvs.recovery.rvs(),
                              self.recovery,
                              'recovery for #{}'.format(self.identifier))
        elif self.immune_status == 'recovered':
            pass
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

        # Use resampling to get a death age > current age.
        while True:
            death_age = self.rvs.mortality.rvs()
            if death_age > age:
                break
        self.events['mortality'] = event.Event(self.birth_date + death_age,
                                               self.mortality,
                                               'mortality for #{}'.format(
                                                   self.identifier))

        if self.sex == 'female':
            self.events['give_birth'] \
              = event.Event(self.herd.time
                            + self.rvs.birth.rvs(self.herd.time, age),
                            self.give_birth,
                            'give birth for #{}'.format(self.identifier))

    def age(self):
        return self.herd.time - self.birth_date

    def mortality(self):
        self.herd.mortality(self)

    def give_birth(self):
        if self.immune_status == 'recovered':
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'

        self.herd.birth(immune_status = calf_status)
        self.events['give_birth'] \
          = event.Event(
              self.herd.time
              + self.rvs.birth.rvs(self.herd.time, self.age()),
              self.give_birth,
              'give birth for #{}'.format(self.identifier))

    def maternal_immunity_waning(self):
        assert self.immune_status == 'maternal immunity'
        self.immune_status = 'susceptible'
        try:
            del self.events['maternal_immunity_waning']
        except KeyError:
            pass

    def infection(self):
        assert self.is_susceptible()
        self.immune_status = 'infectious'
        try:
            del self.events['infection']
        except KeyError:
            pass
        
        self.events['recovery'] \
          = event.Event(self.herd.time
                        + self.rvs.recovery.rvs(),
                        self.recovery,
                        'recovery for #{}'.format(self.identifier))
    
    def recovery(self):
        assert self.is_infectious()
        self.immune_status = 'recovered'
        try:
            del self.events['recovery']
        except KeyError:
            pass
    
    def get_next_event(self):
        return min(self.events.values())

    def is_susceptible(self):
        return self.immune_status == 'susceptible'

    def is_infectious(self):
        return self.immune_status == 'infectious'

    ## Fix me! ##
    def update_infection_time(self, force_of_infection):
        if self.is_susceptible():
            if (force_of_infection > 0.):
                infection_time = stats.expon.rvs(
                    scale = 1. / force_of_infection)
            
                self.events['infection'] = event.Event(
                    self.herd.time + infection_time,
                    self.infection,
                    'infection for #{}'.format(self.identifier))
            else:
                try:
                    del self.events['infection']
                except KeyError:
                    pass
