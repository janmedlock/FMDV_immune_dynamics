from herd.parameters import Parameters
from herd import birth
from herd import chronic_recovery
from herd import chronic_transmission_rate
from herd import female
from herd import immunity_waning
from herd import initial_conditions
from herd import maternal_immunity_waning
from herd import mortality
from herd import probability_chronic
from herd import progression
from herd import recovery
from herd import transmission_rate


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
        self.initial_conditions = initial_conditions.gen(params)

    def __repr__(self):
        'Make instances print nicely.'
        # Like `Parameters()`, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        params_clsname = '{}.{}'.format(Parameters.__module__,
                                        Parameters.__name__)
        return repr(self.parameters).replace(params_clsname, clsname)
