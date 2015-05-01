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
        self.R0 = 5.
        self.birthSeasonalVariance = 1.
        self.initialInfections = 2
        self.probabilityOfMaleBirth = 0.5
        self.maternalImmunityDuration = 0.5
        self.populationSize = 100
        self.infectionDuration = 1.7 / 365.

    def __repr__(self):
        return repr(self.__dict__)


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
