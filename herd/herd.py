import numpy
import collections

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
        self.by_immune_status = collections.defaultdict(list)
        self.identifier = 0

        # Loop until we get a non-zero number of initial infections.
        while True:
            status_ages = self.rvs.endemic_equilibrium.rvs(
                self.params.population_size)
            if len(status_ages['infectious']) > 0:
                break
            else:
                print(
                    'Initial infections = 0!  Re-sampling initial conditions.')

        for (immune_status, ages) in status_ages.items():
            for age in ages:
                if self.debug:
                    if age == 0:
                        print('t = {}: birth of #{} with status {}'.format(
                            self.time,
                            self.identifier,
                            immune_status))
                    else:
                        print('t = {}: arrival of #{} at age {} with status {}'.format(
                            self.time,
                            self.identifier,
                            age,
                            immune_status))

                self.append(buffalo.Buffalo(self, immune_status, age,
                                            building_herd = True))

    def update_infection_times(self):
        number_infectious_new = len(self.by_immune_status['infectious'])

        if ((not hasattr(self, 'number_infectious'))
            or (number_infectious_new != self.number_infectious)):
            self.number_infectious = number_infectious_new

            for b in self.by_immune_status['susceptible']:
                b.update_infection_time()

    def get_stats(self):
        stats = [len(self.by_immune_status[status])
                 for status in ('maternal immunity', 'susceptible',
                                'infectious', 'recovered')]

        return [self.time, stats]

    def get_next_event(self):
        if len(self) > 0:
            # Consider storing all events for the herd in an efficient
            # data type to avoid looping through the whole list.
            return min(b.get_next_event() for b in self)
        else:
            return None

    def stop(self):
        return (self.number_infectious == 0)

    def step(self, tmax = numpy.inf):
        self.update_infection_times()
        event = self.get_next_event()

        if (event is not None) and (event.time < tmax):
            if self.debug:
                print('t = {}: {} for buffalo #{}'.format(
                    event.time,
                    event.__module__,
                    event.buffalo.identifier))
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

        if self.run_number is not None:
            t_last = result[-1][0]
            print('Simulation #{} ended at {:g} days.'.format(self.run_number,
                                                              365 * t_last))

        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
