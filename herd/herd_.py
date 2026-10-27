import itertools

import numpy
import pandas

from . import buffalo, events, immune_status, parameters, random_variables


class Herd(set):
    '''A herd of buffaloes, the things that can happen to them, and code to
    simulate.'''
    def __init__(self, params=None, debug=False, seed=None,
                 _max_samples=100):
        rng = numpy.random.default_rng(seed)
        if params is None:
            params = parameters.Parameters(seed=rng)
        self.params = params
        self.debug = debug
        self.rvs = random_variables.RandomVariables(self.params, seed=rng)
        self.time = 0
        self.events = events.Herd()
        self.by_immune_status = {s: set()
                                 for s in immune_status.immune_statuses}
        self.identifiers = itertools.count(0)
        # These need to be defined before initializing
        # susceptible `buffalo.Buffalo()`.
        self.number_infectious = self.number_carrier = 0
        # Sample until there are some infected hosts.
        for _ in range(_max_samples):
            age_immune_status = self.rvs.initial_conditions.rvs()
            is_infected = age_immune_status.isin(
                {'exposed', 'infectious', 'carrier'}
            )
            if is_infected.any() > 0:
                break
            else:
                print('No infected hosts! Resampling!')
        else:
            raise RuntimeError(f'No infected hosts after {_max_samples=}!')
        for (age, immune_status_) in age_immune_status.items():
            self.add(buffalo.Buffalo(self, immune_status_, age,
                                     initializing_herd=True))

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
        needs_update = False
        number_infectious_new = len(self.by_immune_status['infectious'])
        if number_infectious_new != self.number_infectious:
            self.number_infectious = number_infectious_new
            needs_update = True
        number_carrier_new = len(self.by_immune_status['carrier'])
        if number_carrier_new != self.number_carrier:
            self.number_carrier = number_carrier_new
            needs_update = True
        if needs_update:
            infections = self.events.infections
            for infection in infections:
                infection.time = infection.sample_time()
            self.events.infections = infections

    def get_stats(self):
        stats = [len(self.by_immune_status[s])
                 for s in immune_status.immune_statuses]
        return (self.time, stats)

    @property
    def number_infected(self):
        return sum(len(self.by_immune_status[s])
                   for s in immune_status.infected)

    def stop(self):
        return (self.number_infected == 0)

    def step(self, tmax=numpy.inf):
        self.update_infection_times()
        try:
            event = next(self.events)
        except StopIteration:  # No events
            pass
        else:
            if event.time <= tmax:
                if self.debug:
                    print(event)
                # Make sure the simulation is moving *forward* in time.
                assert ((event.time >= self.time)
                        | numpy.isclose(event.time, self.time))
                self.time = event.time
                event()
                return
        # There are no events or `event.time > tmax`
        self.time = tmax

    def run(self, tmax, print_duration=False, logging_prefix=None,
            _to_dataframe=True):
        result = []
        while True:
            result.append(self.get_stats())
            if (self.time == tmax) or self.stop():
                break
            self.step(tmax)
        if print_duration:
            msg_prefix = ('' if logging_prefix is None
                          else logging_prefix + ': ')
            t_last = result[-1][0]
            print(msg_prefix + f'ended after {t_last:g} year')
        if not _to_dataframe:
            return result
        result = pandas.DataFrame.from_dict(
            dict(result), orient='index',
            columns=immune_status.immune_statuses)
        result.index.name = 'time'
        result.columns.name = 'status'
        return result

    def find_extinction_time(self, tmax):
        result = self.run(tmax, _to_dataframe=False)
        return result[-1][0]
