from collections import defaultdict
from itertools import chain, count

import numpy
from pandas import DataFrame
from sortedcontainers import SortedSet

from herd.buffalo import Buffalo
from herd import event
from herd.parameters import Parameters
from herd.random_variables import RandomVariables


statuses = ('maternal immunity', 'susceptible', 'exposed',
            'infectious', 'chronic', 'recovered')


class HerdEvents(SortedSet):
    '''Container to hold all events that can happen in the herd.
    The `Event()`s are sorted by their time so that the
    one with minimum time can be found efficiently.
    `Infection()` events are also stored in the `infections` attribute,
    since they need to be updated frequently.'''
    def __init__(self):
        super().__init__(key=self.key)
        self.infections = set()

    @staticmethod
    def key(event_):
        return event_.time

    def add(self, value):
        super().add(value)
        if isinstance(value, event.Infection):
            self.infections.add(value)

    def update(self, *iterables):
        super().update(*iterables)
        for value in chain(*iterables):
            if isinstance(value, event.Infection):
                self.infections.add(value)

    def remove(self, value):
        super().remove(value)
        if isinstance(value, event.Infection):
            self.infections.remove(value)

    def get_next(self):
        try:
            return self[0]
        except IndexError:
            return None

    def update_infection_times(self):
        '''Update the infection events.'''
        for infection in self.infections:
            # Use the `super()` versions here
            # to avoid remove() and add() to `infections`.
            super().remove(infection)
            infection.time = infection.sample_time()
            super().add(infection)


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
        self.immune_status_groups = defaultdict(set)
        self.identifiers = count(0)
        status_ages = self.rvs.initial_conditions.rvs(
            self.params.population_size)
        # These need to be defined before initializing
        # susceptible `Buffalo()`.
        self.number_infectious = self.number_chronic = 0
        for (immune_status, ages) in status_ages.items():
            for age in ages:
                self.add(Buffalo(self, immune_status, age))

    def immune_status_add(self, b):
        self.immune_status_groups[b.immune_status].add(b)

    def immune_status_remove(self, b):
        self.immune_status_groups[b.immune_status].remove(b)

    def add(self, b):
        super().add(b)
        self.immune_status_add(b)

    def remove(self, b):
        self.immune_status_remove(b)
        super().remove(b)

    def update_infection_times(self):
        number_infectious_new = len(self.immune_status_groups['infectious'])
        number_chronic_new = len(self.immune_status_groups['chronic'])
        updated = False
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            updated = True
        if number_chronic_new != self.number_chronic:
            self.number_chronic = number_chronic_new
            if self.rvs.chronic_transmission_rate > 0:
                updated = True
        if updated:
            self.events.update_infection_times()

    def get_stats(self):
        stats = [len(self.immune_status_groups[status])
                 for status in statuses]
        return (self.time, stats)

    @property
    def number_infected(self):
        return (len(self.immune_status_groups['exposed'])
                + len(self.immune_status_groups['infectious'])
                + len(self.immune_status_groups['chronic']))

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
        while (self.time < self.params.start_time + tmax):
            self.step(tmax)
            result.append(self.get_stats())
            if self.stop():
                break
        if self.run_number is not None:
            t_last = result[-1][0]
            print('{}Simulation #{} ended after {:g} days.'.format(
                self.logging_prefix, self.run_number,
                365 * (t_last - self.params.start_time)))
        result = DataFrame(dict(result), index=statuses).T
        result.index.name = 'time (d)'
        result.columns.name = 'status'
        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
