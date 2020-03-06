from itertools import count

import numpy
from pandas import DataFrame

from herd.buffalo import Buffalo
from herd.events import HerdEvents
from herd.immune_statuses import immune_statuses
from herd.parameters import Parameters
from herd.random_variables import RandomVariables


class Herd(set):
    '''A herd of buffaloes, the things that can happen to them, and code to
    simulate.'''
    def __init__(self, params=None, debug=False, run_number=None, seed=None,
                 logging_prefix=''):
        if params is None:
            params = Parameters()
        self.params = params
        self.debug = debug
        self.run_number = run_number
        if (seed is None) and (run_number is not None):
            seed = run_number
        if seed is not None:
            numpy.random.seed(seed)
        self.logging_prefix = logging_prefix
        self.rvs = RandomVariables(self.params)
        self.time = self.params.start_time
        self.events = HerdEvents()
        self.by_immune_status = {s: set() for s in immune_statuses}
        self.identifiers = count(0)
        # These need to be defined before initializing
        # susceptible `Buffalo()`.
        self.number_infectious = self.number_chronic = 0
        # Sample until there are some infected animals.
        while True:
            immune_status_ages = self.rvs.initial_conditions.rvs()
            infected = sum(len(immune_status_ages[s])
                           for s in {'exposed', 'infectious', 'chronic'})
            if infected > 0:
                break
            else:
                print('No infected animals! Resampling!')
        for (immune_status, ages) in immune_status_ages.items():
            for age in ages:
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
        '''Calculate the number of infectious and chronic buffalo,
        then, if these are different than previous, update
        the infection times for susceptible buffalo.'''
        updated = False
        number_infectious_new = len(self.by_immune_status['infectious'])
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            updated = True
        if self.rvs.chronic_transmission_rate > 0:
            # This only matters if the chronic transmission rate is non-zero.
            number_chronic_new = len(self.by_immune_status['chronic'])
            if number_chronic_new != self.number_chronic:
                self.number_chronic = number_chronic_new
                updated = True
        if updated:
            self.events.update_infection_times()

    def get_stats(self):
        stats = [len(self.by_immune_status[s])
                 for s in immune_statuses]
        return (self.time, stats)

    @property
    def number_infected(self):
        return sum(len(self.by_immune_status[s])
                   for s in {'exposed', 'infectious', 'chronic'})

    def stop(self):
        return (self.number_infected == 0)

    def step(self, tmax=numpy.inf):
        self.update_infection_times()
        event = self.events.get_next()
        if ((event is not None)
            and (event.time < self.params.start_time + tmax)):
            if self.debug:
                print(event)
            self.time = event.time
            event()
        else:
            self.time = self.params.start_time + tmax

    def run(self, tmax):
        result = [self.get_stats()]
        while self.time < self.params.start_time + tmax:
            self.step(tmax)
            result.append(self.get_stats())
            if self.stop():
                break
        if self.run_number is not None:
            t_last = result[-1][0]
            print('{}simulation #{} ended after {:g} days.'.format(
                self.logging_prefix, self.run_number,
                365 * (t_last - self.params.start_time)))
        result = DataFrame.from_dict(dict(result),
                                     orient='index',
                                     columns=immune_statuses)
        result.index.name = 'time (y)'
        result.columns.name = 'status'
        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
