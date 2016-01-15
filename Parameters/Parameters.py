from .birth import *
from .endemicEquilibrium import *
from .male import *
from .maternalImmunity import *
from .mortality import *
from .recovery import *
from .transmission import *


class Parameters(object):
    def __init__(self):
        'Initialize with default values.'
        self.R0 = 10.
        self.birthSeasonalVariance = 1.
        self.probabilityOfMaleBirth = 0.5
        self.maternalImmunityDuration = 0.5
        self.populationSize = 1000
        self.infectionDuration = 21. / 365.

    def __repr__(self):
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        paramreprs = ['{!r}: {!r}'.format(k, self.__dict__[k])
                      for k in sorted(self.__dict__.keys())]
        return '<{}: {{{}}}>'.format(clsname, ', '.join(paramreprs))


class RandomVariables(object):
    def __init__(self, parameters = None):
        if parameters is None:
            parameters = Parameters()
        self.parameters = parameters

        self.male = male_gen(parameters)

        self.maternalImmunityWaning = maternalImmunityWaning_gen(parameters)

        self.mortality = mortality_gen(parameters)

        self.recovery = recovery_gen(parameters)

        self.transmissionRate = transmissionRate_gen(parameters)

        self.birth = birth_gen(parameters)

        self.endemicEquilibrium = endemicEquilibrium_gen(parameters)

    def __repr__(self):
        # Like Parameters, but with module & class changed.
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        params_clsname = '{}.{}'.format(Parameters.__module__,
                                        Parameters.__name__)

        return repr(self.parameters).replace(params_clsname, clsname)
