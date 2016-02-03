from scipy import stats

from . import rv


class gen(rv.RV):
    '''
    Exponential waiting time to recovery with rate
    recovery_infection_duration.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.infection_duration = parameters.recovery_infection_duration

        distn = stats.expon(scale = self.infection_duration)
        super()._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('infection_duration', ))
