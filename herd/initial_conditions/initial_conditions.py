import numpy
import pandas
from scipy.stats import multinomial

from herd import age_structure, parameters
from herd.initial_conditions import infection, state_probabilities


class gen:
    '''This assumes that all newborns have maternal antibodies (M).'''
    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call rvs() repeatedly.
        self.age_structureRV = age_structure.gen(self.parameters)
        self.hazard_infection = infection.find_hazard(
            self.parameters)

    def _M_prob(self, age):
        '''Probability of being in M, i.e. having maternal immunity.'''
        return state_probabilities.M_prob(age,
                                          self.hazard_infection,
                                          self.parameters)

    def _S_prob(self, age):
        '''Probability of being in S, i.e. susceptible.'''
        return state_probabilities.S_prob(age,
                                          self.hazard_infection,
                                          self.parameters)

    def _C_prob(self, age):
        '''Probability of being in C, i.e. chronically infected.'''
        return state_probabilities.C_prob(age,
                                          self.hazard_infection,
                                          self.parameters)

    def _L_prob(self, age):
        '''Probability of being in L, i.e. having reduced antibodies.'''
        return state_probabilities.L_prob(age,
                                          self.hazard_infection,
                                          self.parameters)

    def _proportion(self, age):
        if numpy.ndim(age) == 0:
            age = numpy.array([age])
        rows = pandas.Index(age, name='age')
        status = pandas.DataFrame(index=rows)
        status['maternal immunity'] = self._M_prob(age)
        status['susceptible'] = self._S_prob(age)
        status['exposed'] = 0
        status['infectious'] = 0
        status['chronic'] = self._C_prob(age)
        status['lost immunity'] = self._L_prob(age)
        # The remaining proportion are recovered.
        # Sum over the statuses.
        not_R = status.sum(axis=(status.ndim - 1))
        status['recovered'] = 1 - not_R
        assert numpy.all(status >= 0)
        assert numpy.all(status <= 1)
        return status

    def rvs(self, size=None):
        if size is None:
            size = self.parameters.initial_infectious
        assert (size >= self.parameters.initial_infectious)
        # Loop until we get a satisfactory sample.
        while True:
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
        # `self._proportion(age) * self.age_structureRV.pdf(age)`
        # but broadcast the multiplication across rows.
        return self._proportion(age).mul(self.age_structureRV.pdf(age),
                                         axis='index')
