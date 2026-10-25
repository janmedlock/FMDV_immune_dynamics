'''Events that can happen to a buffalo.'''

from abc import ABC, abstractmethod

import numpy

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


class SampleWaitingTimeFromAgeMixin:
    def sample_waiting_time_from_age(self, rv, max_samples=10_000):
        '''Get a sample waiting_time from a sample age that is at least
        the buffalo's current age.'''
        for _ in range(max_samples):
            age = rv.rvs()
            if age >= self.buffalo.age:
                return age - self.buffalo.age
        raise ValueError(f'Resampling failed after {max_samples=}!')


class Mortality(SampleWaitingTimeFromAgeMixin, Event):
    '''A buffalo dying.'''
    def is_valid(self):
        return True  # All buffalo can die.

    def do(self):
        self.buffalo.die()

    def sample_waiting_time(self):
        return self.sample_waiting_time_from_age(
            self.buffalo.herd.rvs.mortality)


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
        return self.buffalo.herd.rvs.birth.rvs(self.buffalo.herd.time,
                                               self.buffalo.age)


class MaternalImmunityWaning(SampleWaitingTimeFromAgeMixin, Event):
    '''A buffalo losing maternal immunity.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'maternal_immunity'

    def do(self):
        self.buffalo.change_immune_status_to('susceptible')

    def sample_waiting_time(self):
        return self.sample_waiting_time_from_age(
            self.buffalo.herd.rvs.maternal_immunity_waning)

events_by_immune_status['maternal_immunity'].add(MaternalImmunityWaning)


class Infection(Event):
    '''Base class for a buffalo becoming infected.'''
    def do(self):
        self.buffalo.change_immune_status_to('exposed')

    def force_of_infection(self):
        return ((self.buffalo.herd.rvs.transmission_rate
                 * self.buffalo.herd.number_infectious)
                + (self.buffalo.herd.rvs.carrier_transmission_rate
                   * self.buffalo.herd.number_carrier))

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.infection.rvs(
            self.force_of_infection())


class SusceptibleInfection(Infection):
    '''A buffalo becoming infected from 'susceptible'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'susceptible'

    def do(self):
        self.buffalo.change_immune_status_to('exposed')

events_by_immune_status['susceptible'].add(SusceptibleInfection)


class Progression(Event):
    '''A buffalo becoming infectious.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'exposed'

    def do(self):
        self.buffalo.change_immune_status_to('infectious')

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.progression.rvs()

events_by_immune_status['exposed'].add(Progression)


class Recovery(Event):
    '''A buffalo recovering from acute infection.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'infectious'

    def do(self):
        carrier = self.buffalo.herd.rvs.probability_carrier.rvs()
        if carrier:
            self.buffalo.change_immune_status_to('carrier')
        else:
            self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.recovery.rvs()

events_by_immune_status['infectious'].add(Recovery)


class CarrierRecovery(Event):
    '''A buffalo recovering from carrier infected.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'carrier'

    def do(self):
        self.buffalo.change_immune_status_to('recovered_true')

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.carrier_recovery.rvs()

events_by_immune_status['carrier'].add(CarrierRecovery)


class LostImmunityInfection(Infection):
    '''Base class for a buffalo becoming infected from
    'lost_immunity_true' or 'recovered_false'.'''
    def force_of_infection(self):
        return (self.buffalo.herd.params.lost_immunity_susceptibility
                * super().force_of_infection())


class LostImmunityTrueInfection(LostImmunityInfection):
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


class RecoveredFalseInfection(LostImmunityInfection):
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


class AntibodyLoss(Event):
    '''Base class for a buffalo losing apparent antibodies.'''
    def do(self):
        lost_immunity_true \
            = self.buffalo.herd.rvs.probability_lost_immunity_true.rvs()
        if lost_immunity_true:
            self.buffalo.change_immune_status_to('lost_immunity_true')
        else:
            self.buffalo.change_immune_status_to('lost_immunity_false')

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.antibody_loss.rvs()


class RecoveredTrueAntibodyLoss(AntibodyLoss):
    '''A buffalo losing apparent antibodies from 'recovered_true'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'recovered_true'

events_by_immune_status['recovered_true'].add(RecoveredTrueAntibodyLoss)


class RecoveredFalseAntibodyLoss(AntibodyLoss):
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


class AntibodyGain(Event):
    '''Base class for a buffalo gaining apparent antibodies.'''
    def do(self):
        recovered_true = self.buffalo.herd.rvs.probability_recovered_true.rvs()
        if recovered_true:
            self.buffalo.change_immune_status_to('recovered_true')
        else:
            self.buffalo.change_immune_status_to('recovered_false')

    def sample_waiting_time(self):
        return self.buffalo.herd.rvs.antibody_gain.rvs()


class LostImmunityTrueAntibodyGain(AntibodyGain):
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


class LostImmunityFalseAntibodyGain(AntibodyGain):
    '''A buffalo gaining apparent antibodies from 'lost_immunity_false'.'''
    def is_valid(self):
        return self.buffalo.immune_status == 'lost_immunity_false'

events_by_immune_status['lost_immunity_false'].add(
    LostImmunityFalseAntibodyGain
)
