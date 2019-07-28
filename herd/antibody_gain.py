from scipy.stats import expon

from herd.rv import RV


class gen(RV):
    '''Waiting time to gain of antibodies.'''
    def __init__(self, parameters, *args, **kwargs):
        self.antibody_gain_hazard = parameters.antibody_gain_hazard
        scale = 1 / self.antibody_gain_hazard
        distn = expon(scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('antibody_gain_hazard', ))
