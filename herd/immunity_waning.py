from scipy.stats import expon

from herd.rv import RV


class gen(RV):
    '''Exponential waiting time until loss of immunity from
    recovered to susceptible.'''
    def __init__(self, parameters, *args, **kwargs):
        self.immunity_waning_duration = parameters.immunity_waning_duration
        distn = expon(scale=self.immunity_waning_duration)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('immunity_waning_duration', ))
