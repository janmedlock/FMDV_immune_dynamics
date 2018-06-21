from numpy import asarray
from scipy.stats import bernoulli

from herd import age_structure, maternal_immunity_waning


class gen:
    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call rvs() repeatedly.
        self.age_structureRV = age_structure.gen(self.parameters)
        self.maternal_immunity_waningRV = maternal_immunity_waning.gen(
            self.parameters)

    def rvs(self, size=1):
        ages = self.age_structureRV.rvs(size=size)
        status_ages = {}
        # Who has maternal antibodies.
        prob_M = self.maternal_immunity_waningRV.sf(ages)
        is_M = asarray(bernoulli.rvs(prob_M), dtype=bool)
        status_ages['maternal immunity'] = ages[is_M]
        ages = ages[~is_M]
        # Figure out who is S and who is C/R.
        status_ages['rest'] = ages
        # Figure out who is E/I.
        return status_ages
