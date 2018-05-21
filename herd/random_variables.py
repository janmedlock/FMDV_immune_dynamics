from . import parameters
from . import birth
from . import male
from . import maternal_immunity_waning
from . import mortality
from . import progression
from . import recovery
from . import transmission_rate
from . import chronic_transmission_rate
from . import probability_chronic
from . import chronic_recovery
from . import immunity_waning


class RandomVariables(object):
    def __init__(self, params = None):
        if params is None:
            params = parameters.Parameters()
        self.parameters = params
        self.male = male.gen(params)
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

    def __repr__(self):
        'Make instances print nicely.'
        # Like Parameters, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        params_clsname = '{}.{}'.format(parameters.Parameters.__module__,
                                        parameters.Parameters.__name__)
        return repr(self.parameters).replace(params_clsname, clsname)
