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
        self._initialized = False

        self.R0 = 5.
        self.birthSeasonalVariance = 1.
        self.initialInfections = 2
        self.probabilityOfMaleBirth = 0.5
        self.maternalImmunityDuration = 0.5
        self.populationSize = 100
        self.infectionDuration = 1.7 / 365.

        self.build()
        self._initialized = True

    def get_male(self):
        self.male = male_gen(self.probabilityOfMaleBirth)

    def get_maternalImmunityWaning(self):
        self.maternalImmunityWaning = maternalImmunityWaning_gen(
            self.maternalImmunityDuration)

    def get_mortality(self):
        self.mortality = mortality_gen()

    def get_recovery(self):
        self.recovery = recovery_gen(self.infectionDuration)

    def get_transmissionRate(self):
        self.transmissionRate = transmissionRate_gen(self.R0,
                                                     self.recovery,
                                                     self.populationSize)

    def get_birth(self):
        self.birth = birth_gen(self.mortality,
                               self.male,
                               self.birthSeasonalVariance)

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
        self._set('probabilityOfMaleBirth', p)

        if self._initialized:
            self.get_male()
            self.get_birth()
            self.get_ageStructure()

    def set_maternalImmunityDuration(self, d):
        # self.maternalImmunityDuration = d
        self._set('maternalImmunityDuration', d)

        if self._initialized:
            self.get_maternalImmunityWaning()

    def set_infectionDuration(self, d):
        # self.infectionDuration = d
        self._set('infectionDuration', d)

        if self._initialized:
            self.get_recovery()
            self.get_transmissionRate()

    def set_R0(self, r):
        # self.R0 = r
        self._set('R0', r)
        
        if self._initialized:
            self.get_transmissionRate()

    def set_populationSize(self, p):
        # self.populationSize = p
        self._set('populationSize', p)

        if self._initialized:
            self.get_transmissionRate()

    def set_birthSeasonalVariance(self, a):
        # self.birthSeasonalVariance = a
        self._set('birthSeasonalVariance', a)

        if self._initialized:
            self.get_birth()
            self.get_ageStructure()

    def _set(self, k, v):
        'Really set the value, bypassing self.__setattr__()'
        super(Parameters, self).__setattr__(k, v)

    def __setattr__(self, k, v):
        'Try to use set_* methods if available.'
        try:
            set_k = getattr(self, 'set_{}'.format(k))
            set_k(v)
        except (AttributeError, TypeError):
            self._set(k, v)
