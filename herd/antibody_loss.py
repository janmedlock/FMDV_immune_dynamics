import numpy
from scipy.stats import expon

from . import rv


class RV(rv.RV):
    '''Waiting time to loss of antibodies is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_loss_hazard = parameters.antibody_loss_hazard
        assert self.antibody_loss_hazard >= 0
        if self.antibody_loss_hazard > 0:
            scale = 1 / self.antibody_loss_hazard
        else:  # self.antibody_loss_hazard = 0
            scale = numpy.inf
        distn = expon(scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('antibody_loss_hazard', ))
