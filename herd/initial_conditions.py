import numpy
import pandas
from scipy.stats import multinomial

from herd import age_structure, maternal_immunity_waning
from herd import _initial_conditions


class gen:
    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call rvs() repeatedly.
        self.age_structureRV = age_structure.gen(self.parameters)
        self.maternal_immunity_waningRV = maternal_immunity_waning.gen(
            self.parameters)
        h = _initial_conditions.find_hazard_infection()
        self.hazard_infection = h['Pooled']

    def _proportion(self, age):
        status = {}
        total = 0
        # M.
        p = self.maternal_immunity_waningRV.sf(age)
        status['maternal immunity'] = p
        total += p
        # S.
        p = _initial_conditions.S_prob(age, self.hazard_infection,
                                       self.maternal_immunity_waningRV)
        status['susceptible'] = p
        total += p
        # R.
        # Hopefully this is just fixing roundoff errors.
        total = numpy.clip(total, 0, 1)
        status['recovered'] = 1 - total
        # FIX ME: Figure out who is E/I/C.
        # Force the same order as in the `status` dict.
        cols = status.keys()
        if numpy.isscalar(age):
            return pandas.Series(status, index=cols)
        else:
            rows = pandas.Index(age, name='age')
            return pandas.DataFrame(status, index=rows, columns=cols)

    def rvs(self, size=1):
        # Pick `size` random ages.
        ages = self.age_structureRV.rvs(size=size)
        # Determine the status for each age.
        proportions = self._proportion(ages)
        status = proportions.columns
        status_ages = {k: [] for k in status}
        # `scipy.stats.multinomial.rvs()` can't handle multiple `p`s,
        # so we need to loop.
        for (age, row) in proportions.iterrows():
            # Randomly pick a status.
            rv = multinomial.rvs(1, row)
            # `rv` is an array with `1` in the position
            # picked and `0`s in the remaining positions.
            # Convert that to the name.
            s = status[rv == 1][0]
            # Add this `age` to the status list.
            status_ages[s].append(age)
        return status_ages

    def pdf(self, age):
        # `self._proportion(age) * self.age_structureRV.pdf(age)`
        # but broadcast the multiplication across rows.
        return self._proportion(age).mul(self.age_structureRV.pdf(age),
                                         axis='index')
