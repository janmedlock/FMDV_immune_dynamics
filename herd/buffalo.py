from scipy import stats

from . import events

# NOTE: Need to update transmission rate from carriers
#(requires herd to calculate number_chronic)

class Buffalo:
    'A single buffalo and the actions that can occur to it.'

    def __init__(self, herd, immune_status = 'maternal immunity', age = 0,
                 building_herd = False):
        self.herd = herd
        self.immune_status = immune_status

        self.birth_date = self.herd.time - age

        self.identifier = next(self.herd.identifiers) # like an iterator?

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

        self.events = events.Events(self)

        self.set_mortality()

        if self.immune_status == 'maternal immunity':
            self.set_maternal_immunity_waning()
        elif self.immune_status == 'susceptible':
            # Defer if building_herd because
            # self.herd.number_infectious won't be correct.
            if not building_herd:
                self.update_infection()
        elif self.immune_status == 'exposed':
            self.set_progression()
        elif self.immune_status == 'infectious':
            self.set_recovery_or_progression()
        elif self.immune_status == 'chronic':
            self.set_chronic_progression()
        elif self.immune_status == 'recovered':
            self.set_immunity_waning()
        else:
            raise ValueError('Unknown immune_status = {}!'.format(
                self.immune_status))

        self.set_sex()

        if self.sex == 'female':
            self.update_give_birth()

    def age(self):
        return (self.herd.time - self.birth_date)

    def is_susceptible(self):
        return (self.immune_status == 'susceptible')

    def is_exposed(self):
        return (self.immune_status == 'exposed')

    def is_infectious(self):
        return (self.immune_status == 'infectious')

    def is_chronic(self):   # NEW
        return (self.immune_status == 'chronic')

    def is_recovered(self):
        return (self.immune_status == 'recovered')

    def set_sex(self):
        self.sex = 'male' if (self.herd.rvs.male.rvs() == 1) else 'female'

    def change_immune_status_to(self, new_immune_status):
        self.herd.immune_status_remove(self)
        self.immune_status = new_immune_status
        self.herd.immune_status_append(self)

    def mortality(self):
        self.herd.remove(self)

    def set_mortality(self):
        # Use resampling to get a mortality time > current time.
        while True:
            mortality_time = self.birth_date + self.herd.rvs.mortality.rvs()
            if mortality_time > self.herd.time:
                break
        self.events.add('mortality', mortality_time)

    def give_birth(self):
        if self.immune_status == 'recovered':
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'
        self.herd.append(Buffalo(self.herd, calf_status))
        self.update_give_birth()

    def update_give_birth(self):
        self.events.update('give_birth',
                           self.herd.time
                           + self.herd.rvs.birth.rvs(self.herd.time,
                                                     self.age()))

    def maternal_immunity_waning(self):
        assert self.immune_status == 'maternal immunity'
        self.change_immune_status_to('susceptible')
        self.update_infection()
        self.events.remove('maternal_immunity_waning')

    def set_maternal_immunity_waning(self):
        while True:
            waning_time = (self.birth_date
                           + self.herd.rvs.maternal_immunity_waning.rvs())
            if waning_time >= self.herd.time:
                break
        self.events.add('maternal_immunity_waning',
                        waning_time)

    def progression(self):
        assert self.is_exposed()
        self.change_immune_status_to('infectious')
        self.events.remove('progression')
        self.set_recovery_or_progression()

    def set_recovery_or_progression(self):
        # CHECK IF COIN FLIP HAPPENS EVERY TIME OR ONCE?
        if (self.herd.rvs.probability_chronic.rvs() == 1):
            self.set_chronic_progression()
        else:
            self.set_recovery()

    def set_progression(self):
        self.events.add('progression',
                        self.herd.time + self.herd.rvs.progression.rvs())

    def recovery(self):
        assert self.is_infectious()
        self.change_immune_status_to('recovered')
        self.events.remove('recovery')
        self.set_immunity_waning()

    def set_recovery(self):
        self.events.add('recovery',
                        self.herd.time + self.herd.rvs.recovery.rvs())

    def chronic_progression(self):
        assert self.is_infectious()
        self.change_immune_status_to('chronic')
        self.events.remove('chronic')
        self.set_chronic_recovery()

    def set_chronic_progression(self):
        self.events.add('chronic_progression',
                        self.herd.time + self.herd.rvs.recovery.rvs())

    def immunity_waning(self):
        assert self.is_recovered()
        self.change_immune_status_to('susceptible')
        self.events.remove('immunity_waning')
        self.update_infection()
        print('WARNING: Immunity waning')

    def set_immunity_waning(self):
        self.events.add('immunity_waning',
                        self.herd.time + self.herd.rvs.immunity_waning.rvs())

    def chronic_recovery(self):
        assert self.is_chronic()
        self.change_immune_status_to('recovered')
        self.events.remove('chronic_recovery')
        self.set_chronic_recovery()

    def set_chronic_recovery(self):
        self.events.add('chronic_recovery',
                        self.herd.time + self.herd.rvs.chronic_recovery.rvs())

    ###################################################################

    def infection(self):
        assert self.is_susceptible()
        self.change_immune_status_to('exposed')
        self.events.remove('infection')
        self.set_progression()

    # Fix me!
    def update_infection(self):
        assert self.is_susceptible()

        if (self.herd.number_infectious > 0):
            force_of_infection = (self.herd.rvs.transmission_rate
                                  * self.herd.number_infectious
                                  + self.herd.rvs.chronic_transmission_rate
                                  * self.herd.number_chronic)

            infection_time = self.herd.time + stats.expon.rvs(
                scale = 1 / force_of_infection)

            self.events.update('infection', infection_time)
        else:
            try:
                self.events.remove('infection')
            except KeyError:
                pass

    def get_next_event(self):
        # Consider storing the events in a data type that's more
        # efficient to find the minimum.
        return self.events.get_next()
