from scipy.stats import multinomial

from herd import age_structure
from herd.initial_conditions import immune_status


class gen:
    '''This assumes that all newborns have maternal antibodies (M)
    and that the hazard of infection is constant in time,
    not periodic as it is with a periodic birth pulse.'''

    def __init__(self, parameters):
        self.parameters = parameters
        # Reuse these in case we call pdf() or rvs() repeatedly.
        self.pdf_interpolant = immune_status.probability_interpolant(
            self.parameters)
        self.ages = age_structure.gen(self.parameters)

    def rvs(self, size=None):
        if size is None:
            size = self.parameters.population_size
        # Pick `size` random ages.
        ages = self.ages.rvs(size=size)
        # Determine the immune status for each age.
        immune_status_probability = self.pdf(ages)
        immune_statuses = immune_status_probability.columns
        immune_status_ages = {k: [] for k in immune_statuses}
        # `scipy.stats.multinomial.rvs()` can't handle multiple `p`s,
        # so we need to loop.
        for (age, row) in immune_status_probability.iterrows():
            # Randomly pick an immune status.
            rv = multinomial.rvs(1, row / row.sum())
            # `rv` is an array with `1` in the position
            # picked and `0`s in the remaining positions.
            # Convert that to the name.
            s = immune_statuses[rv == 1][0]
            # Add this `age` to the status list.
            immune_status_ages[s].append(age)
        return immune_status_ages

    def pdf(self, age):
        return self.pdf_interpolant(age)
