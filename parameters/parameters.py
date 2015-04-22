from .ageStructure import *
from .birth import *
from .male import *
from .maternalImmunity import *
from .mortality import *
from .recovery import *
from .transmission import *


class Parameters(object):
    def __init__(self):
        'Initialize with default values.'

        self.R0 = 5.
        self.birthSeasonalAmplitude = 1.
        self.initialInfections = 2
        self.probabilityOfMaleBirth = 0.5
        self.maternalImmunityDuration = 0.5
        self.populationSize = 100
        self.infectionDuration = 1.6 / 365.

        self.build()

    def get_male(self):
        self.male = male_gen(self.probabilityOfMaleBirth)

    def get_maternalImmunityWaning(self):
        self.maternalImmunityWaning = maternalImmunityWaning_gen(
            self.maternalImmunityDuration)

    def get_mortality(self):
        self.mortality = mortality_gen(name = 'mortality', a = 0.)

    def get_recovery(self):
        self.recovery = recovery_gen(self.infectionDuration)

    def get_transmissionRate(self):
        self.transmissionRate = transmissionRate_gen(self.R0,
                                                     self.recovery,
                                                     self.populationSize)

    def get_birth(self):
        self.birth = birth_gen(self.mortality,
                               self.male,
                               self.birthSeasonalAmplitude,
                               name = 'birth',
                               a = 0.,
                               shapes = 'time0, age0')

    def get_ageStructure(self):
        self.ageStructure = ageStructure_gen(self.mortality,
                                             self.birth,
                                             self.male)

    def build(self):
        self.get_male()
        self.get_maternalImmunityWaning()
        self.get_mortality()
        self.get_recovery()
        self.get_transmissionRate()
        self.get_birth()
        self.get_ageStructure()

    def set_probabilityOfMaleBirth(self, p):
        # self.probabilityOfMaleBirth = p
        object.__setattr__(self, 'probabilityOfMaleBirth', p)

        self.get_male()
        self.get_birth()
        self.get_ageStructure()

    def set_maternalImmunityDuration(self, d):
        # self.maternalImmunityDuration = d
        object.__setattr__(self, 'maternalImmunityDuration', d)

        self.get_maternalImmunityWaning()

    def set_infectionDuration(self, d):
        # self.infectionDuration = d
        object.__setattr__(self, 'infectionDuration', d)

        self.get_recovery()
        self.get_transmissionRate()

    def set_R0(self, r):
        # self.R0 = r
        object.__setattr__(self, 'R0', r)
        
        self.get_transmissionRate()

    def set_populationSize(self, p):
        # self.populationSize = p
        object.__setattr__(self, 'populationSize', p)

        self.get_transmissionRate()

    def set_birthSeasonalAmplitude(self, a):
        # self.birthSeasonalAmplitude = a
        object.__setattr__(self, 'birthSeasonalAmplitude', a)

        self.get_birth()
        self.get_ageStructure()

    def __setattr__(self, k, v):
        'Try to use set_* methods if available.'
        try:
            f = getattr(self, 'set_{}'.format(k))
            f(v)
        except (AttributeError, TypeError):
            object.__setattr__(self, k, v)
