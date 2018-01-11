from scipy import stats

from . import rv

class gen(rv.RV):
    '''
    Exponential waiting time until loss of immunity from
    recovered to susceptible.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.immunity_waning_duration = parameters.immunity_waning_duration
        distn = stats.expon(scale = self.immunity_waning_duration)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('immunity_waning_duration', ))
