from .ageStructure import *
from .birth_triangular import *
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

        self.male = male_gen(parameters.probabilityOfMaleBirth)

        self.maternalImmunityWaning = maternalImmunityWaning_gen(
            parameters.maternalImmunityDuration)

        self.mortality = mortality_gen()

        self.recovery = recovery_gen(parameters.infectionDuration)

        self.transmissionRate = transmissionRate_gen(parameters.R0,
                                                     self.recovery,
                                                     parameters.populationSize)

        self.birth = birth_gen(self.mortality,
                               self.male,
                               parameters.birthSeasonalVariance)

        self.ageStructure = ageStructure_gen(self.mortality,
                                             self.birth,
                                             self.male)
