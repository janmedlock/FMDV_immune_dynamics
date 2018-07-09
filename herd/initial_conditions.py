import numpy
import pandas
from scipy.stats import multinomial

from herd import age_structure, maternal_immunity_waning


class gen:
    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call rvs() repeatedly.
        self.age_structureRV = age_structure.gen(self.parameters)
        self.maternal_immunity_waningRV = maternal_immunity_waning.gen(
            self.parameters)

    def _proportion(self, age):
        status = {}
        # Who has maternal antibodies.
        status['maternal immunity'] = self.maternal_immunity_waningRV.sf(age)
        # Figure out who is S and who is C/R.
        status['rest'] = 1 - status['maternal immunity']
        # Figure out who is E/I.
        if numpy.isscalar(age):
            return pandas.Series(status)
        else:
            idx = pandas.Index(age, name='age')
            return pandas.DataFrame(status, index=idx)

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
