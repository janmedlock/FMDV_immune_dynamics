from scipy import stats

from . import rv

class gen(rv.RV):
    'Whether an offspring is male is a Bernoulli RV.'

    def __init__(self, parameters, *args, **kwargs):
        self.probability_chronic = parameters.probability_chronic

        distn = stats.bernoulli(self.probability_chronic)
        super()._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('probability_chronic', ))
