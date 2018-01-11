from scipy import stats

from . import rv

class gen(rv.RV):
    'Whether an offspring is male is a Bernoulli RV.'

    def __init__(self, parameters, *args, **kwargs):
        self.male_probability_at_birth = parameters.male_probability_at_birth

        distn = stats.bernoulli(self.male_probability_at_birth)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('male_probability_at_birth', ))
