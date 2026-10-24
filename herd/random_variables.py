from . import (antibody_gain, antibody_loss, birth, carrier_recovery,
               carrier_transmission_rate, female, initial_conditions,
               maternal_immunity_waning, mortality, probability_carrier,
               probability_lost_immunity_true, probability_recovered_true,
               progression, recovery, transmission_rate)
from .parameters import Parameters


class RandomVariables:
    def __init__(self, params=None):
        if params is None:
            params = Parameters()
        self.parameters = params
        self.female = female.gen(params)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(params)
        self.mortality = mortality.gen(params)
        self.progression = progression.gen(params)
        self.recovery = recovery.gen(params)
        self.transmission_rate = transmission_rate.gen(params)
        self.carrier_transmission_rate = carrier_transmission_rate.gen(params)
        self.birth = birth.gen(params)
        self.probability_carrier = probability_carrier.gen(params)
        self.carrier_recovery = carrier_recovery.gen(params)
        self.antibody_loss = antibody_loss.gen(params)
        self.probability_lost_immunity_true \
            = probability_lost_immunity_true.gen(params)
        self.antibody_gain = antibody_gain.gen(params)
        self.probability_recovered_true \
            = probability_recovered_true.gen(params)
        self.initial_conditions = initial_conditions.gen(params)

    def __repr__(self):
        'Make instances print nicely.'
        # Like `Parameters()`, but with module & class changed.
        params_clsname = f'{Parameters.__module__}.{Parameters.__name__}'
        clsname = f'{self.__module__}.{self.__class__.__name__}'
        return repr(self.parameters).replace(params_clsname, clsname)
