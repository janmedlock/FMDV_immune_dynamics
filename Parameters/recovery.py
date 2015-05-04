from scipy import stats

from . import rv


class recovery_gen(rv.RV):
    'Exponential waiting time to recovery with rate infectionDuration.'

    def __init__(self, parameters, *args, **kwargs):
        self.infectionDuration = parameters.infectionDuration

        distn = stats.expon(scale = self.infectionDuration)
        super(recovery_gen, self)._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('infectionDuration', ))
