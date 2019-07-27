from scipy.stats import expon

from herd.rv import RV


class gen(RV):
    '''Waiting time to loss of antibodies.'''
    def __init__(self, parameters, *args, **kwargs):
        # FIXME
        self.immunity_waning_duration = parameters.immunity_waning_duration
        # FIXME
        distn = expon(scale=self.immunity_waning_duration)
        super()._copyattrs(distn)

    def __repr__(self):
        # FIXME
        return super().__repr__(('immunity_waning_duration', ))
