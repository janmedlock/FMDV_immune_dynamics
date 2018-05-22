from scipy import stats

from . import age_structure
from . import maternal_immunity_waning


class _RVs:
    def __init__(self, parameters):
        self.age_structure = age_structure.gen(parameters)
        self.maternal_immunity_waning = maternal_immunity_waning.gen(parameters)


class gen:
    def __init__(self, parameters):
        self.parameters = parameters
        self.RVs = _RVs(self.parameters)

    def rvs(self, size=1):
        ages = self.RVs.age_structure.rvs(size=size)
        status_ages = {}
        # Who has maternal antibodies.
        prob_M = self.RVs.maternal_immunity_waning.sf(ages)
        is_M = stats.bernoulli.rvs(prob_M).astype(bool)
        status_ages['maternal immunity'] = ages[is_M]
        ages = ages[~is_M]
        # Figure out who is S and who is C/R.
        status_ages['rest'] = ages
        # Figure out who is E/I.
        return status_ages
