from .R0 import *
from .recovery import *
from .populationSize import *


def findTransmissionRate(R0, recovery, populationSize):
    return R0 / recovery.mean() / populationSize


transmissionRate = findTransmissionRate(R0, recovery, populationSize)
