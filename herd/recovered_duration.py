from scipy import stats

from . import rv

class gen(rv.RV):
    '''
    Exponential waiting time until loss of immunity from 
    recovered to susceptible.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.recovered_duration = parameters.recovered_duration
        distn = stats.expon(scale = self.recovered_duration)

    def __repr__(self):
        return rv.RV.__repr__(self, ('recovered_duration', ))
