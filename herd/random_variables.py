from . import parameters
from . import birth
from . import endemic_equilibrium
from . import male
from . import maternal_immunity_waning
from . import mortality
from . import recovery


class RandomVariables(object):
    def __init__(self, params = None):
        if params is None:
            params = parameters.Parameters()
        self.parameters = params

        self.male = male.gen(params)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(params)
        self.mortality = mortality.gen(params)
        self.recovery = recovery.gen(params)
        self.birth = birth.gen(params)
        self.endemic_equilibrium = endemic_equilibrium.gen(params)

    def __repr__(self):
        'Make instances print nicely.'

        # Like Parameters, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        params_clsname = '{}.{}'.format(parameters.Parameters.__module__,
                                        parameters.Parameters.__name__)

        return repr(self.parameters).replace(params_clsname, clsname)
