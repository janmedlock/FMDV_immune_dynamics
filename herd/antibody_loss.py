import numpy
from scipy.stats import expon

from herd.rv import RV


class gen(RV):
    '''Waiting time to loss of antibodies.'''
    def __init__(self, parameters, *args, **kwargs):
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
