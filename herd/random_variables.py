from . import parameters
from . import birth
from . import endemic_equilibrium
from . import male
from . import maternal_immunity_waning
from . import mortality
from . import recovery
from . import transmission_rate


class RandomVariables(object):
    def __init__(self, parameters = None):
        if parameters is None:
            parameters = parameters.Parameters()
        self.parameters = parameters

        self.male = male.gen(parameters)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(parameters)
        self.mortality = mortality.gen(parameters)
        self.recovery = recovery.gen(parameters)
        self.transmission_rate = transmission_rate.gen(parameters)
        self.birth = birth.gen(parameters)
        self.endemic_equilibrium = endemic_equilibrium.gen(parameters)

    def __repr__(self):
        'Make instances print nicely.'

        # Like Parameters, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        params_clsname = '{}.{}'.format(parameters.Parameters.__module__,
                                        parameters.Parameters.__name__)

        return repr(self.parameters).replace(params_clsname, clsname)
