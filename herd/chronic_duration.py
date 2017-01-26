from scipy import stats

from . import rv
#is_exp_duration = True

class gen(rv.RV):
    '''
    Exponential waiting time to recovery from chronic state with 1/rate
    recovery_chronic_duration.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.chronic_duration = parameters.chronic_duration
        distn = stats.expon(scale = self.chronic_duration)

    def __repr__(self):
        return rv.RV.__repr__(self, ('chronic_duration', ))
