from scipy import stats

from . import rv

class gen(rv.RV):
    '''
    Exponential waiting time until loss of immunity from 
    recovered to susceptible.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.immunity_waning = parameters.immunity_waning
        distn = stats.expon(scale = self.immunity_waning)

    def __repr__(self):
        return rv.RV.__repr__(self, ('immunity_waning', ))
