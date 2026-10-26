'''Events that can happen to a buffalo.'''

from abc import ABC, abstractmethod

from .immune_statuses import immune_statuses

# Events that can happen to each immune status.
events_by_immune_status = {
    immune_status: set()
    for immune_status in immune_statuses
}


class Event(ABC):
    '''Parent class for events that happen to a buffalo.'''
    @abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event.
        Subclasses must define this method.'''

    @abstractmethod
    def do(self):
        '''Execute the event.
        Subclasses must define this method.'''

    @abstractmethod
    def sample_waiting_time(self):
        '''Generate a random sample waiting time for the event.
        Subclasses must define this method.'''

    def __init__(self, buffalo):
        self.buffalo = buffalo
        assert self.is_valid()
        self.time = self.sample_time()

    def __call__(self):
        assert self.is_valid()
        self.buffalo.events.remove(self)
        self.do()

    @staticmethod
    def sample_waiting_time_from(rv, *args, **kwds):
        '''Helper for sample_waiting_time.'''
        return rv.rvs(*args, **kwds)

    def sample_time(self):
        '''Generate a random sample time for the event.'''
        return self.buffalo.herd.time + self.sample_waiting_time()

    def remove_event(self, cls):
        for e in self.buffalo.events:
            if isinstance(e, cls):
                self.buffalo.events.remove(e)
                break
        else:
            raise RuntimeError(f'Missing {cls.__name__}() event!')

    def __repr__(self):
        return (
            f't={self.time:g}: '
            + self.__class__.__name__
            + f' of #{self.buffalo.identifier}'
        )


class _AgeDependent:
    '''A mixin for `Event()` with waiting times dependent on only age.'''
    def sample_age(self, rv, max_samples=10_000):
        '''Get a sample age that is at least the buffalo's current age.'''
        for _ in range(max_samples):
            age = rv.rvs()
            if age >= self.buffalo.age:
                return age
        raise ValueError(f'Resampling failed after {max_samples=}!')

    def sample_waiting_time_from(self, rv):
        if self.buffalo.age == 0:
            return rv.rvs()
        return self.sample_age(rv) - self.buffalo.age


class _TimeSinceEntryDependent:
    '''A mixin for `Event()` with waiting times dependent on only time
    since entry.'''
    # TODO


class Birth(Event):
    '''A buffalo giving birth.'''
    def is_valid(self):
        return self.buffalo.sex == 'female'

    def do(self):
        self.buffalo.give_birth()
        # Add next birth.
        self.time = self.sample_time()
        self.buffalo.events.add(self)

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.birth,
            self.buffalo.herd.time, self.buffalo.age)


class Mortality(_AgeDependent, Event):
    '''A buffalo dying.'''
    def is_valid(self):
        return True  # All buffalo can die.

    def do(self):
        self.buffalo.die()

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.mortality)


class MaternalImmunityWaning(_AgeDependent, Event):
    '''A buffalo losing maternal immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'maternal_immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.maternal_immunity_waning)

events_by_immune_status['maternal_immunity'].add(MaternalImmunityWaning)


class Progression(_TimeSinceEntryDependent, Event):
    '''A buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.progression)

events_by_immune_status['exposed'].add(Progression)


class Recovery(_TimeSinceEntryDependent, Event):
    '''A buffalo recovering from acute infection.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'infectious'

    def do(self):
        if self.buffalo.herd.rvs.recovery_to_carrier.rvs():
            self.buffalo.change_immune_status_to('carrier')
        else:
            self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.recovery)

events_by_immune_status['infectious'].add(Recovery)


class CarrierRecovery(_TimeSinceEntryDependent, Event):
    '''A buffalo recovering from carrier infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'carrier'

    def do(self):
        self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.carrier_recovery)

events_by_immune_status['carrier'].add(CarrierRecovery)


class _AntibodyLoss(Event):
    '''Base class for a buffalo losing apparent antibodies.'''
    def do(self):
        if self.buffalo.herd.rvs.antibody_loss_to_lost_immunity_true.rvs():
            self.buffalo.change_immune_status_to('lost_immunity_true')
        else:
            self.buffalo.change_immune_status_to('lost_immunity_false')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.antibody_loss)


class RecoveredTrueAntibodyLoss(_AntibodyLoss):
    '''A buffalo losing apparent antibodies from 'recovered_true'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered_true'

events_by_immune_status['recovered_true'].add(RecoveredTrueAntibodyLoss)


class RecoveredFalseAntibodyLoss(_AntibodyLoss):
    '''A buffalo losing apparent antibodies from 'recovered_false'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered_false'

    def do(self):
        # Immune status 'recovered_false' can change by either
        # `RecoveredFalseInfection()` or `RecoveredFalseAntibodyLoss()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(RecoveredFalseInfection)
        super().do()

events_by_immune_status['recovered_false'].add(RecoveredFalseAntibodyLoss)


class _AntibodyGain(Event):
    '''Base class for a buffalo gaining apparent antibodies.'''
    def do(self):
        if self.buffalo.herd.rvs.antibody_gain_to_recovered_true.rvs():
            self.buffalo.change_immune_status_to('recovered_true')
        else:
            self.buffalo.change_immune_status_to('recovered_false')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.antibody_gain)


class LostImmunityTrueAntibodyGain(_AntibodyGain):
    '''A buffalo gaining apparent antibodies from 'lost_immunity_true'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'lost_immunity_true'

    def do(self):
        # Immune status 'lost_immunity_true' can change by either
        # `LostImmunityTrueInfection()` or `LostImmunityTrueAntibodyGain()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(LostImmunityTrueInfection)
        super().do()

events_by_immune_status['lost_immunity_true'].add(LostImmunityTrueAntibodyGain)


class LostImmunityFalseAntibodyGain(_AntibodyGain):
    '''A buffalo gaining apparent antibodies from 'lost_immunity_false'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'lost_immunity_false'

events_by_immune_status['lost_immunity_false'].add(
    LostImmunityFalseAntibodyGain
)


class Infection(Event):
    '''Base class for a buffalo becoming infected.'''
    def __init__(self, buffalo, susceptibility=1):
        self.susceptibility = susceptibility
        super().__init__(buffalo)

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.infection,
            self.buffalo.herd.number_infectious,
            self.buffalo.herd.number_carrier,
            self.susceptibility)


class SusceptibleInfection(Infection):
    '''A buffalo becoming infected from 'susceptible'.'''
    def __init__(self, buffalo):
        susceptibility = 1
        super().__init__(buffalo, susceptibility)

    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

events_by_immune_status['susceptible'].add(SusceptibleInfection)


class _LostImmunityInfection(Infection):
    '''Base class for a buffalo becoming infected from
    'lost_immunity_true' or 'recovered_false'.'''
    def __init__(self, buffalo):
        susceptibility = buffalo.herd.params.lost_immunity_susceptibility
        super().__init__(buffalo, susceptibility)


class LostImmunityTrueInfection(_LostImmunityInfection):
    '''A buffalo becoming infected from 'lost_immunity_true'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'lost_immunity_true'

    def do(self):
        # Immune status 'lost_immunity_true' can change by either
        # `LostImmunityTrueInfection()` or `LostImmunityTrueAntibodyGain()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(LostImmunityTrueAntibodyGain)
        super().do()

events_by_immune_status['lost_immunity_true'].add(LostImmunityTrueInfection)


class RecoveredFalseInfection(_LostImmunityInfection):
    '''A buffalo becoming infected from 'recovered_false'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered_false'

    def do(self):
        # Immune status 'recovered_false' can change by either
        # `RecoveredFalseInfection()` or `RecoveredFalseAntibodyLoss()`.
        # Remove the one that didn't happen so it doesn't stay
        # in the queue.
        self.remove_event(RecoveredFalseAntibodyLoss)
        super().do()

events_by_immune_status['recovered_false'].add(RecoveredFalseInfection)
