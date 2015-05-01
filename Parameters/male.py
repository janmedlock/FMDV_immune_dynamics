from scipy import stats

from . import rv

class male_gen(rv.RV):
    'Whether an offspring is male is a Bernoulli RV.'

    def __init__(self, parameters, *args, **kwargs):
        self.probabilityOfMaleBirth = parameters.probabilityOfMaleBirth

        distn = stats.bernoulli(self.probabilityOfMaleBirth)
        super(male_gen, self)._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('probabilityOfMaleBirth', ))
