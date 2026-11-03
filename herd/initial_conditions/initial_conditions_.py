'''Initial conditions.'''

import numpy

from .. import _rv, age, immune_status
from . import _probability


class InitialConditions(_rv.RV):
    '''This assumes that the hazard of infection is constant in time,
    not periodic as it is with a periodic birth pulse.'''

    name = 'initial_conditions'

    def __init__(self, parameters, seed=None, debug=False):
        super().__init__(parameters, seed=seed)
        self.population_size = parameters.population_size
        # Reuse these in case we call pdf() or rvs() repeatedly.
        # TODO: Fix
        if True:
            import warnings

            import pandas

            def immune_status_conditional_pdf(ages):
                pdf = pandas.DataFrame(0.,
                                       index=ages,
                                       columns=immune_status.immune_statuses)
                pdf['infectious'] = 0.05
                pdf['susceptible'] = 1. - pdf.sum(axis='columns')
                return pdf

            warnings.warn('Fake `initial_conditions`!')
            self.immune_status_pdf = immune_status_conditional_pdf
        else:
            self.immune_status_pdf = _probability.Interpolant(parameters,
                                                              debug=debug)
        self.age = age.Age(parameters, seed=self.rng)

    def immune_status_conditional_pdf(self, ages):
        '''The probability of being in each immune state vs. `ages`,
        conditioned on being alive.'''
        p = self.immune_status_pdf(ages)
        return p.divide(p.sum(axis='columns'), axis='index')

    def pdf(self, age_):
        return self.immune_status_conditional_pdf(age_).multiply(
            self.age.pdf(age_), axis='index')

    def rvs(self, size=None):
        if size is None:
            size = self.population_size
        # Pick `size` random ages.
        ages = self.age.rvs(size=size)
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
        immune_statuses = immune_status_probability.columns
        return immune_status_probability.agg(
            lambda p: self.rng.choice(immune_statuses, p=p),
            axis='columns'
        ).astype(immune_status.dtype)


class HerdMixin:
    '''Mixin for '..herd_.Herd()`.'''

    def sample_initial_conditions(self, max_samples=100):
        '''Get a sample initial condition with some infected hosts.'''
        initial_conditions = getattr(self.rvs, InitialConditions.name)

        def has_infected(ser):
            return ser.isin(immune_status.infected).any()

        return _rv.sample_until(initial_conditions.rvs,
                                has_infected,
                                max_samples=max_samples)

