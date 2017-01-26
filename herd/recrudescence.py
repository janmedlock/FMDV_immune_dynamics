from scipy import stats

from . import rv

class gen(rv.RV):
    '''
    Exponential waiting time to recrudesce from the chronic class
    with rate, 1/params.recrudescence.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.recrudescence = parameters.recrudescence
        distn = stats.expon(scale = self.recrudescence)
        super()._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('recovered_duration', ))
