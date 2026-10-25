import numpy

from . import (antibody_gain, antibody_loss, birth, carrier_recovery,
               carrier_transmission_rate, female, infection,
               initial_conditions, maternal_immunity_waning, mortality,
               probability_carrier, probability_lost_immunity_true,
               probability_recovered_true, progression, recovery,
               transmission_rate)
from .parameters import Parameters


class RandomVariables:
    def __init__(self, params=None, seed=None):
        rng = numpy.random.default_rng(seed)
        if params is None:
            params = Parameters(seed=rng)
        self.parameters = params
        self.antibody_gain = antibody_gain.gen(params, seed=rng)
        self.antibody_loss = antibody_loss.gen(params, seed=rng)
        self.birth = birth.gen(params, seed=rng)
        self.carrier_recovery = carrier_recovery.gen(params, seed=rng)
        self.carrier_transmission_rate = carrier_transmission_rate.gen(
            params, seed=rng)
        self.female = female.gen(params, seed=rng)
        self.infection = infection.gen(params, seed=rng)
        self.initial_conditions = initial_conditions.gen(params, seed=rng)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(params,
                                                                     seed=rng)
        self.mortality = mortality.gen(params, seed=rng)
        self.probability_carrier = probability_carrier.gen(params, seed=rng)
        self.probability_lost_immunity_true \
            = probability_lost_immunity_true.gen(params, seed=rng)
        self.probability_recovered_true \
            = probability_recovered_true.gen(params, seed=rng)
        self.progression = progression.gen(params, seed=rng)
        self.recovery = recovery.gen(params, seed=rng)
        self.transmission_rate = transmission_rate.gen(params, seed=rng)

    def __repr__(self):
        'Make instances print nicely.'
        # Like `Parameters()`, but with module & class changed.
        params_clsname = f'{Parameters.__module__}.{Parameters.__name__}'
        clsname = f'{self.__module__}.{self.__class__.__name__}'
        return repr(self.parameters).replace(params_clsname, clsname)
