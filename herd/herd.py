import collections
import itertools

import numpy

from . import buffalo
from . import parameters
from . import random_variables


class Herd(list):
    '''
    A herd of buffaloes, the things that can happen to them, and code to
    simulate.
    '''

    def __init__(self, params = None, debug = False, run_number = None):
        if params is None:
            self.params = parameters.Parameters()
        else:
            self.params = params

        self.debug = debug
        self.run_number = run_number

        self.rvs = random_variables.RandomVariables(self.params)

        self.time = self.params.start_time
        self.immune_status_lists = collections.defaultdict(list)
        self.identifiers = itertools.count(0)

        # Loop until we get a non-zero number of initial infections.
        while True:
            status_ages = self.rvs.endemic_equilibrium.rvs(
                self.params.population_size)
            if (len(status_ages['exposed'])
                + len(status_ages['infectious'])
                + len(status_ages['chronic'])) > 0:
                break
            # else:
            #     print(
            #         'Initial infections = 0!  Resampling initial conditions.')
      
        for (immune_status, ages) in status_ages.items():
            for age in ages:
                self.append(buffalo.Buffalo(self, immune_status, age,
                                            building_herd = True))
    
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
        
        if ((not hasattr(self, 'number_infectious'))
            or (number_infectious_new != self.number_infectious)):
            self.number_infectious = number_infectious_new

        if ((not hasattr(self, 'number_chronic'))
            or (number_chronic_new != self.number_chronic)):
            self.number_chronic = number_chronic_new
            
        for b in self.immune_status_lists['susceptible']:
            b.update_infection()

    def get_stats(self):
        stats = [len(self.immune_status_lists[status])
                 for status in ('maternal immunity', 'susceptible',
                                'exposed', 'infectious', 'chronic', 'recovered')]

        return [self.time, stats]

    def get_next_event(self):
        if len(self) > 0:
            # Consider storing all events for the herd in an efficient
            # data type to avoid looping through the whole list.
            return min(b.get_next_event() for b in self)
        else:
            return None
# https://www.programiz.com/python-programming/property
    @property
    def number_infected(self):
        return (len(self.immune_status_lists['exposed'])
                + len(self.immune_status_lists['infectious'])
                + len(self.immune_status_lists['chronic']))  # OK????????

    def stop(self):
        return (self.number_infected == 0)

    def step(self, tmax = numpy.inf):
        self.update_infection_times()
        event = self.get_next_event()

        if (event is not None) and (event.time < tmax):
            if self.debug:
                print(event)
            self.time = event.time
            event()
        else:
            self.time = tmax

    def run(self, tmax):
        result = [self.get_stats()]

        while (self.time < tmax):
            self.step(tmax)
            result.append(self.get_stats())
            if self.stop():
                break
            ######### Test for carriers!, NEW ##########
            if self.number_chronic > 0:
                print("error, some became chronic carriers")
            
        if self.run_number is not None:
            t_last = result[-1][0]
            print('Simulation #{} ended at {:g} days.'.format(self.run_number,
                                                              365 * t_last))

        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
