import numpy


class RV:
    '''Time to infection is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        # The transmission rates are scaled so that R_0 is constant as
        # initial population size varies
        self.carrier_transmission_rate = (parameters.carrier_transmission_rate
                                          / parameters.population_size)
        self.transmission_rate = (parameters.transmission_rate
                                  / parameters.population_size)
        self.rng = numpy.random.default_rng(seed)

    def rvs(self, number_infectious, number_carrier, susceptibility=1,
            size=None):
        force_of_infection = (susceptibility
                * (self.transmission_rate * number_infectious
                   + self.carrier_transmission_rate * number_carrier))
        if force_of_infection == 0:
            return numpy.inf
        return self.rng.exponential(1 / force_of_infection)
