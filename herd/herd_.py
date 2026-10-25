from itertools import count

import numpy
from pandas import DataFrame

from .buffalo import Buffalo
from .events import HerdEvents
from .immune_statuses import immune_statuses
from .parameters import Parameters
from .random_variables import RandomVariables


class Herd(set):
    '''A herd of buffaloes, the things that can happen to them, and code to
    simulate.'''
    def __init__(self, params=None, debug=False, seed=None,
                 _max_samples=100):
        if params is None:
            params = Parameters()
        self.params = params
        self.debug = debug
        self.rng = numpy.random.default_rng(seed)
        self.rvs = RandomVariables(self.params)
        self.time = self.params.start_time
        self.events = HerdEvents()
        self.by_immune_status = {s: set() for s in immune_statuses}
        self.identifiers = count(0)
        # These need to be defined before initializing
        # susceptible `Buffalo()`.
        self.number_infectious = self.number_carrier = 0
        # Sample until there are some infected hosts.
        for _ in range(_max_samples):
            age_immune_status = self.rvs.initial_conditions.rvs(
                random_state=self.rng)
            is_infected = age_immune_status.isin(
                {'exposed', 'infectious', 'carrier'}
            )
            if is_infected.any() > 0:
                break
            else:
                print('No infected hosts! Resampling!')
        else:
            raise RuntimeError(f'No infected hosts after {_max_samples=}!')
        for (age, immune_status) in age_immune_status.items():
            self.add(Buffalo(self, immune_status, age))

    def immune_status_add(self, b):
        self.by_immune_status[b.immune_status].add(b)

    def immune_status_remove(self, b):
        self.by_immune_status[b.immune_status].remove(b)

    def add(self, b):
        super().add(b)
        self.immune_status_add(b)

    def remove(self, b):
        self.immune_status_remove(b)
        super().remove(b)

    def update_infection_times(self):
        '''Calculate the number of infectious and carrier buffalo,
        then, if these are different than previous, update
        the infection times for susceptible buffalo.'''
        updated = False
        number_infectious_new = len(self.by_immune_status['infectious'])
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            updated = True
        if self.rvs.carrier_transmission_rate > 0:
            # This only matters if the carrier transmission rate is non-zero.
            number_carrier_new = len(self.by_immune_status['carrier'])
            if number_carrier_new != self.number_carrier:
                self.number_carrier = number_carrier_new
                updated = True
        if updated:
            self.events.update_infection_times()

    def get_stats(self):
        stats = [len(self.by_immune_status[s])
                 for s in immune_statuses]
        return (self.time, stats)

    _INFECTED = frozenset({'exposed', 'infectious', 'carrier'})

    @property
    def number_infected(self):
        return sum(len(self.by_immune_status[s])
                   for s in self._INFECTED)

    def stop(self):
        return (self.number_infected == 0)

    def step(self, tmax=numpy.inf):
        self.update_infection_times()
        event = self.events.get_next()
        if ((event is not None)
            and (event.time < self.params.start_time + tmax)):
            if self.debug:
                print(event)
            # Make sure the simulation is moving *forward* in time.
            assert ((event.time >= self.time)
                    | numpy.isclose(event.time, self.time))
            self.time = event.time
            event()
        else:
            self.time = self.params.start_time + tmax

    def run(self, tmax,
            print_duration=False, logging_prefix=None):
        result = [self.get_stats()]
        while self.time < self.params.start_time + tmax:
            self.step(tmax)
            result.append(self.get_stats())
            if self.stop():
                break
        if print_duration:
            msg_prefix = (
                '' if logging_prefix is None
                else logging_prefix + ': '
            )
            t_last = result[-1][0] - self.params.start_time
            print(msg_prefix + f'ended after {t_last:g} year')
        result = DataFrame.from_dict(dict(result),
                                     orient='index',
                                     columns=immune_statuses)
        result.index.name = 'time'
        result.columns.name = 'status'
        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
