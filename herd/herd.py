from collections import defaultdict
from itertools import count

import numpy
from pandas import DataFrame

from herd.buffalo import Buffalo
from herd import event
from herd.parameters import Parameters
from herd.random_variables import RandomVariables


statuses = ('maternal immunity', 'susceptible', 'exposed',
            'infectious', 'chronic', 'recovered')


class Herd(list):
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
        self.immune_status_lists = defaultdict(list)
        self.identifiers = count(0)
        status_ages = self.rvs.initial_conditions.rvs(
            self.params.population_size)
        # These need to be defined before initializing
        # susceptible `Buffalo()`.
        self.number_infectious = self.number_chronic = 0
        for (immune_status, ages) in status_ages.items():
            for age in ages:
                self.append(Buffalo(self, immune_status, age))

    def immune_status_append(self, b):
        self.immune_status_lists[b.immune_status].append(b)

    def immune_status_remove(self, b):
        self.immune_status_lists[b.immune_status].remove(b)

    def append(self, b):
        super().append(b)
        self.immune_status_append(b)

    def remove(self, b):
        self.immune_status_remove(b)
        super().remove(b)

    def update_infection_times(self):
        number_infectious_new = len(self.immune_status_lists['infectious'])
        number_chronic_new = len(self.immune_status_lists['chronic'])
        updated = False
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            updated = True
        if number_chronic_new != self.number_chronic:
            self.number_chronic = number_chronic_new
            if self.rvs.chronic_transmission_rate > 0:
                updated = True
        if updated:
            for b in self.immune_status_lists['susceptible']:
                b.update_infection()

    def get_stats(self):
        stats = [len(self.immune_status_lists[status])
                 for status in statuses]
        return (self.time, stats)

    def get_next_event(self):
        # Consider storing all events for the herd in an efficient
        # data type to avoid looping through the whole list.
        if len(self) > 0:
            return event.get_next(b.get_next_event() for b in self)
        else:
            return None

    @property
    def number_infected(self):
        return (len(self.immune_status_lists['exposed'])
                + len(self.immune_status_lists['infectious'])
                + len(self.immune_status_lists['chronic']))

    def stop(self):
        return (self.number_infected == 0)

    def step(self, tmax=numpy.inf):
        self.update_infection_times()
        event = self.get_next_event()
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
