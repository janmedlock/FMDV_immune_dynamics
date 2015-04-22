from .deterministic import *


def maternalImmunityWaning_gen(maternalImmunityDuration):
        return deterministic(scale = maternalImmunityDuration)
