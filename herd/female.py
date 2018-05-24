from scipy import stats

from . import rv

class gen(rv.RV):
    'Whether an offspring is female is a Bernoulli RV.'

    def __init__(self, parameters, *args, **kwargs):
        self.female_probability_at_birth \
            = parameters.female_probability_at_birth
        distn = stats.bernoulli(self.female_probability_at_birth)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('female_probability_at_birth', ))
