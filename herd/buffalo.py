import numpy
from scipy import stats

from herd.events import Event, Events


def Sex(buffalo):
    '''The buffalo having its sex determined.'''
    if buffalo.herd.rvs.female.rvs() == 1:
        return 'female'
    else:
        return 'male'


class Mortality(Event):
    '''The buffalo dying.'''
    def is_valid(self):
        return True  # All buffalo can die.

    def do(self):
        self.buffalo.herd.remove(self.buffalo)

    def sample_time(self):
        # Use resampling to get a mortality time > current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.mortality.rvs())
            if time > self.buffalo.herd.time:
                return time


class Birth(Event):
    '''The buffalo giving birth.'''
    # Which classes give mom antibodies that she passes on.
    has_antibodies = ('chronic', 'recovered')

    def is_valid(self):
        return self.buffalo.sex == 'female'

    def do(self):
        if self.buffalo.immune_status in self.has_antibodies:
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'
        self.buffalo.herd.append(Buffalo(self.buffalo.herd, calf_status))
        # Update to time of the next birth.
        self.time = self.sample_time()

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.birth.rvs(self.buffalo.herd.time,
                                                  self.buffalo.age))


class MaternalImmunityWaning(Event):
    '''The buffalo losing maternal immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'maternal immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        # Use resampling to get a waning time > current time.
        while True:
            time = (self.buffalo.birth_date
                    + self.buffalo.herd.rvs.maternal_immunity_waning.rvs())
            if time >= self.buffalo.herd.time:
                return time


class Infection(Event):
    '''The buffalo becoming infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Progression(self.buffalo))

    def sample_time(self):
        force_of_infection = (
            (self.buffalo.herd.rvs.transmission_rate
             * self.buffalo.herd.number_infectious)
            + (self.buffalo.herd.rvs.chronic_transmission_rate
               * self.buffalo.herd.number_chronic))
        # scale = 1 / force_of_infection
        # Handle division by 0.
        scale = numpy.ma.filled(
            numpy.ma.divide(1, force_of_infection),
            numpy.inf)
        return (self.buffalo.herd.time
                + stats.expon.rvs(scale=scale))


class Progression(Event):
    '''The buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Recovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.progression.rvs())


class Recovery(Event):
    '''The buffalo recovering from acute infection.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'infectious'

    def do(self):
        if self.buffalo.herd.rvs.probability_chronic.rvs() == 1:
            self.do_chronic_progression()
        else:
            self.do_recovery()

    def do_recovery(self):
        self.buffalo.change_immune_status_to('recovered')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def do_chronic_progression(self):
        self.buffalo.change_immune_status_to('chronic')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ChronicRecovery(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.recovery.rvs())


class ChronicRecovery(Event):
    '''The buffalo recovering from chronically infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'chronic'

    def do(self):
        self.buffalo.change_immune_status_to('recovered')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(ImmunityWaning(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.chronic_recovery.rvs())


class ImmunityWaning(Event):
    '''The buffalo losing its acquired immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')
        self.buffalo.events.remove(self)
        self.buffalo.events.add(Infection(self.buffalo))

    def sample_time(self):
        return (self.buffalo.herd.time
                + self.buffalo.herd.rvs.immunity_waning.rvs())


class Buffalo:
    '''A single buffalo and the events that can occur to it.'''
    def __init__(self, herd, immune_status='maternal immunity', age=0):
        self.herd = herd
        self.immune_status = immune_status
        self.birth_date = self.herd.time - age
        self.sex = Sex(self)
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
        # Add all valid `Event()` subclasses.
        # for klass in Event.__subclasses__():
        #     try:
        #         event = klass(self)
        #     except AssertionError:
        #         pass
        #     else:
        #         self.events.add(event)
        self.events.add(Mortality(self))
        if self.sex == 'female':
            self.events.add(Birth(self))
        if self.immune_status == 'maternal immunity':
            self.events.add(MaternalImmunityWaning(self))
        elif self.immune_status == 'susceptible':
            # When building a new herd, the infection time
            # won't be correct because
            # `self.herd.number_infectious` won't be finalized.
            self.events.add(Infection(self))
        elif self.immune_status == 'exposed':
            self.events.add(Progression(self))
        elif self.immune_status == 'infectious':
            self.events.add(Recovery(self))
        elif self.immune_status == 'chronic':
            self.events.add(ChronicRecovery(self))
        elif self.immune_status == 'recovered':
            self.events.add(ImmunityWaning(self))
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
        event = self.events['Infection']
        assert event.is_valid()
        event.time = event.sample_time()

    def get_next_event(self):
        # Consider storing the events in a data type that's more
        # efficient to find the minimum.
        return self.events.get_next()
