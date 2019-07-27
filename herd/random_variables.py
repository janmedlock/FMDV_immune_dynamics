from herd.parameters import Parameters
from herd import (antibody_gain, antibody_loss, birth,
                  chronic_recovery, chronic_transmission_rate, female,
                  immunity_waning, initial_conditions,
                  maternal_immunity_waning, mortality, probability_chronic,
                  progression, recovery, transmission_rate)


class RandomVariables:
    def __init__(self, params = None):
        if params is None:
            params = Parameters()
        self.parameters = params
        self.female = female.gen(params)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(params)
        self.mortality = mortality.gen(params)
        self.progression = progression.gen(params)
        # waiting time I to R
        self.recovery = recovery.gen(params)
        self.transmission_rate = transmission_rate.gen(params)
        self.chronic_transmission_rate = chronic_transmission_rate.gen(params)
        self.birth = birth.gen(params)
        # proportion I to C
        self.probability_chronic = probability_chronic.gen(params)
        # waiting time to recover C to R
        self.chronic_recovery = chronic_recovery.gen(params)
        # waiting time leaving R to S
        self.immunity_waning = immunity_waning.gen(params)
        # waiting time leaving R to P
        self.antibody_loss = antibody_loss.gen(params)
        # waiting time leaving P to R
        self.antibody_gain = antibody_gain.gen(params)
        self.initial_conditions = initial_conditions.gen(params)

    def __repr__(self):
        'Make instances print nicely.'
        # Like `Parameters()`, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        params_clsname = '{}.{}'.format(Parameters.__module__,
                                        Parameters.__name__)
        return repr(self.parameters).replace(params_clsname, clsname)
