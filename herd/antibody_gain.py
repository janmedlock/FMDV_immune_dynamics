import numpy
from scipy.stats import expon

from .rv import RV


class gen(RV):
    '''Waiting time to gain of antibodies is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_gain_hazard = parameters.antibody_gain_hazard
        assert self.antibody_gain_hazard >= 0
        if self.antibody_gain_hazard > 0:
            scale = 1 / self.antibody_gain_hazard
        else:  # self.antibody_gain_hazard = 0
            scale = numpy.inf
        distn = expon(scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('antibody_gain_hazard', ))
