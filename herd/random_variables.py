from . import parameters
from . import birth
from . import endemic_equilibrium
from . import male
from . import maternal_immunity_waning
from . import mortality
from . import progression
from . import recovery
from . import transmission_rate
from . import probability_chronic
from . import recrudescence
from . import chronic_duration


class RandomVariables(object):
    def __init__(self, params = None):
        if params is None:
            params = parameters.Parameters()
        self.parameters = params

        self.male = male.gen(params)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(params)
        self.mortality = mortality.gen(params)
        self.progression = progression.gen(params)
        self.recovery = recovery.gen(params)
        self.transmission_rate = transmission_rate.gen(params)
        self.birth = birth.gen(params)
        self.endemic_equilibrium = endemic_equilibrium.gen(params)
        self.probability_chronic = probability_chronic.gen(params)  # prop I to C 
        self.recrudescence = recrudescence.gen(params)  # 1/rate to recrudesce R to S
        self.chronic_duration = chronic_duration.gen(params) # 1/rate to recover C to R

    def __repr__(self):
        'Make instances print nicely.'

        # Like Parameters, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        params_clsname = '{}.{}'.format(parameters.Parameters.__module__,
                                        parameters.Parameters.__name__)

        return repr(self.parameters).replace(params_clsname, clsname)
