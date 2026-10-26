import numpy

from . import (antibody_gain, antibody_gain_to_recovered_true,
               antibody_loss, antibody_loss_to_lost_immunity_true,
               birth, carrier_recovery, female, infection,
               initial_conditions, maternal_immunity_waning, mortality,
               progression, recovery, recovery_to_carrier)
from .parameters import Parameters


class RandomVariables:
    def __init__(self, params=None, seed=None):
        rng = numpy.random.default_rng(seed)
        if params is None:
            params = Parameters(seed=rng)
        self.parameters = params
        self.antibody_gain = antibody_gain.RV(params, seed=rng)
        self.antibody_gain_to_recovered_true \
            = antibody_gain_to_recovered_true.RV(params, seed=rng)
        self.antibody_loss = antibody_loss.RV(params, seed=rng)
        self.antibody_loss_to_lost_immunity_true \
            = antibody_loss_to_lost_immunity_true.RV(params, seed=rng)
        self.birth = birth.RV(params, seed=rng)
        self.carrier_recovery = carrier_recovery.RV(params, seed=rng)
        self.female = female.RV(params, seed=rng)
        self.infection = infection.RV(params, seed=rng)
        self.initial_conditions = initial_conditions.RV(params, seed=rng)
        self.maternal_immunity_waning = maternal_immunity_waning.RV(params,
                                                                    seed=rng)
        self.mortality = mortality.RV(params, seed=rng)
        self.recovery_to_carrier = recovery_to_carrier.RV(params, seed=rng)
        self.progression = progression.RV(params, seed=rng)
        self.recovery = recovery.RV(params, seed=rng)

    def __repr__(self):
        'Make instances print nicely.'
        # Like `Parameters()`, but with module & class changed.
        params_clsname = f'{Parameters.__module__}.{Parameters.__name__}'
        clsname = f'{self.__module__}.{self.__class__.__name__}'
        return repr(self.parameters).replace(params_clsname, clsname)
