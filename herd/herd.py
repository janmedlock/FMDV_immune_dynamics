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

        self.time = parameters.start_time
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
                self.birth(immune_status, age)

    def mortality(self, b):
        self.remove(b)

    def birth(self, immune_status = 'maternal immunity', age = 0):
        if self.debug:
            if age > 0:
                print('t = {}: arrival of #{} at age {} with status {}'.format(
                    self.time,
                    self.identifier,
                    age,
                    immune_status))
            else:
                print('t = {}: birth of #{} with status {}'.format(
                    self.time,
                    self.identifier,
                    immune_status))

        self.append(buffalo.Buffalo(self, immune_status, age,
                                    identifier = self.identifier))
        self.identifier += 1

    def update_infection_times(self):
        self.number_infectious = sum(1 for b in self
                                     if b.is_infectious())
        self.force_of_infection = (self.rvs.transmission_rate
                                   * self.number_infectious)
        for b in self:
            b.update_infection_time(self.force_of_infection)

    def get_stats(self):
        counts = []
        for status in ('maternal immunity', 'susceptible',
                       'infectious', 'recovered'):
            counts.append(sum(1 for b in self
                              if b.immune_status == status))
        return [self.time, counts]

    def get_next_event(self):
        if len(self) > 0:
            return min([b.get_next_event() for b in self])
        else:
            return None

    def stop(self):
        try:
            return (self.number_infectious == 0)
        except AttributeError:
            # If self.number_infectious isn't defined yet.
            return False

    def step(self, tmax = numpy.inf):
        self.update_infection_times()
        event = self.get_next_event()

        if (event is not None) and (event.time < tmax):
            if self.debug:
                print('t = {}: {}'.format(event.time, event.label))
            self.time = event.time
            event()
        else:
            self.time = tmax

    def run(self, tmax):
        result = [self.get_stats()]

        while (self.time < tmax) and (not self.stop()):
            self.step(tmax)
            result.append(self.get_stats())

        if self.run_number is not None:
            t_last = result[-1][0]
            print('Simulation #{} ended at {:g} days.'.format(self.run_number,
                                                              365 * t_last))

        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax)
        return result[-1][0]
