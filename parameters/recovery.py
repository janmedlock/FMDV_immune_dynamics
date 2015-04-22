from .deterministic import *


def recovery_gen(infectionDuration):
    return deterministic(scale = infectionDuration)
