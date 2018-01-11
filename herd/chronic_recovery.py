from scipy import stats

from . import rv


class gen(rv.RV):
    '''
    Exponential waiting time to recovery from chronic state with rate
    1 / chronic_recovery.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.chronic_recovery = parameters.chronic_recovery
        distn = stats.expon(scale = self.chronic_recovery)

    def __repr__(self):
        return super().__repr__(('chronic_recovery', ))
