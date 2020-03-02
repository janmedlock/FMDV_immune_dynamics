import numpy
import pandas
from scipy.stats import multinomial

from herd import age_structure, parameters
from herd.initial_conditions import infection, status


class gen:
    '''This assumes that all newborns have maternal antibodies (M)
    and that the hazard of infection is constant in time,
    not periodic as it is with a periodic birth pulse.'''

    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call rvs() repeatedly.
        self.age_structureRV = age_structure.gen(self.parameters)
        self.hazard_infection = infection.find_hazard(
            self.parameters)

    def _status_probability(self, age):
        return status.probability(age,
                                  self.hazard_infection,
                                  self.parameters)

    def rvs(self, size=None):
        if size is None:
            size = self.parameters.initial_infectious
        assert (size >= self.parameters.initial_infectious)
        # Loop until we get a satisfactory sample.
        while True:
            # Pick `size` random ages.
            ages = self.age_structureRV.rvs(size=size)
            # Determine the status for each age.
            status_probability = self._status_probability(ages)
            statuses = status_probability.columns
            status_ages = {k: [] for k in statuses}
            # `scipy.stats.multinomial.rvs()` can't handle multiple `p`s,
            # so we need to loop.
            for (age, row) in status_probability.iterrows():
                # Randomly pick a status.
                rv = multinomial.rvs(1, row)
                # `rv` is an array with `1` in the position
                # picked and `0`s in the remaining positions.
                # Convert that to the name.
                s = statuses[rv == 1][0]
                # Add this `age` to the status list.
                status_ages[s].append(age)
            if (len(status_ages['susceptible'])
                < self.parameters.initial_infectious):
                # We don't have enough susceptibles.  Loop again.
                continue
            else:
                # Convert a few susceptibles to infectious.
                for _ in range(self.parameters.initial_infectious):
                    age = status_ages['susceptible'].pop()
                    status_ages['infectious'].append(age)
                # This is a satisfactory sample, so end loop.
                break
        return status_ages

    def pdf(self, age):
        # `self._status_probability(age) * self.age_structureRV.pdf(age)`
        # but broadcast the multiplication across rows (statuses).
        return self._status_probability(age).mul(self.age_structureRV.pdf(age),
                                                 axis='index')
