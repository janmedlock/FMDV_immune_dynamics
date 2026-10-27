'''Initial conditions.'''

import numpy
import pandas

from .. import age_structure
from . import immune_status


class RV:
    '''This assumes that the hazard of infection is constant in time,
    not periodic as it is with a periodic birth pulse.'''

    def __init__(self, parameters, seed=None, debug=False):
        self.parameters = parameters
        self.rng = numpy.random.default_rng(seed)
        # Reuse these in case we call pdf() or rvs() repeatedly.
        # TODO: Fix
        if True:
            import warnings

            import pandas

            from ..immune_status import immune_statuses

            def immune_status_conditional_pdf(ages):
                pdf = pandas.DataFrame(0.,
                                       index=ages,
                                       columns=immune_statuses)
                pdf['infectious'] = 0.05
                pdf['susceptible'] = 1. - pdf.sum(axis='columns')
                return pdf

            warnings.warn('Fake `initial_conditions`!')
            self.immune_status_pdf = immune_status_conditional_pdf
        else:
            self.immune_status_pdf = (
                immune_status.probability.Interpolant(self.parameters,
                                                      debug=debug))
        self.ages = age_structure.RV(self.parameters, seed=self.rng)

    def immune_status_conditional_pdf(self, ages):
        '''The probability of being in each immune state vs. `ages`,
        conditioned on being alive.'''
        p = self.immune_status_pdf(ages)
        return p.divide(p.sum(axis='columns'), axis='index')

    def pdf(self, age):
        return self.immune_status_conditional_pdf(age).multiply(
            self.ages.pdf(age), axis='index')

    def rvs(self, size=None):
        if size is None:
            size = self.parameters.population_size
        # Pick `size` random ages.
        ages = self.ages.rvs(size=size)
        # Determine the immune status for each age.
        immune_status_probability = self.immune_status_conditional_pdf(ages)
        # `immune_status_probability.sum(axis='columns')`
        # might be slightly different than 1 due to roundoff
        # errors, which breaks `rng.choice()`.
        while True:
            totals = immune_status_probability.sum(axis='columns')
            mask = ~numpy.isclose(totals, 1)
            if not mask.any():
                break
            immune_status_probability[mask] = (
                immune_status_probability[mask].divide(totals[mask],
                                                       axis='index')
            )
        # Randomly pick an immune status for each row.
        immune_status_ = immune_status_probability.columns
        dtype = pandas.CategoricalDtype(immune_status_, ordered=True)
        return immune_status_probability.agg(
            lambda p: self.rng.choice(immune_status_, p=p),
            axis='columns'
        ).astype(dtype)
