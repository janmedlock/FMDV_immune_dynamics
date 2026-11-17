'''Handle waiting times.'''

import numpy

from .. import events


def name(waiting_time):
    module = waiting_time.__module__
    return module.split('.')[-1]


class BirthWaitingTime:
    '''Avoid the slow `..floquet.find_birth_scaling()`'''

    def __init__(self, parameters, mortality):
        self.birth_menarche = parameters.birth_menarche
        self.female_probability = parameters.female_probability
        self.mortality = mortality

    def hazard(self, ages):
        hazard = events.birth.hazard_no_seasonality(
            self.birth_menarche, self.female_probability,
            ages, _scaling=1)
        # Scale to get 0 population growth rate using the `ages`
        # from `Solver()`, rather than the ages used by
        # `.._floquet.find_birth_scaling()`
        (da, *da_rest) = numpy.diff(ages)
        assert numpy.allclose(da, da_rest)
        hazard /= (
            self.female_probability
            * hazard
            * self.mortality.sf(ages)
        ).sum() * da
        return hazard



class InfectionWaitingTime:
    '''Dummy infection waiting time with hazard = 1.'''
    def __init__(self, parameters):
        pass

    def hazard(self, ages):
        return numpy.ones_like(ages, dtype=float)


class WaitingTimes:
    '''Helper to get parameterized instances of waiting times.'''

    def __init__(self, parameters):
        self.parameters = parameters
        self._d = {}  # Cache

    def __getitem__(self, waiting_time):
        try:
            return self._d[waiting_time]
        except KeyError:
            pass
        if waiting_time is events.birth.WaitingTime:
            val = BirthWaitingTime(self.parameters,
                                   self[events.mortality.WaitingTime])
        elif waiting_time is events.infection.WaitingTime:
            val = InfectionWaitingTime(self.parameters)
        else:
            val = waiting_time(self.parameters)
        self._d[waiting_time] = val
        return val
