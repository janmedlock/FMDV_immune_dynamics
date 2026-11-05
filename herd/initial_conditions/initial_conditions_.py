'''Initial conditions.'''

import warnings

import numpy
import pandas

from .. import _rv, age, immune_status
from . import _interpolate, _solver


class InitialConditions(_rv.RV):
    '''This assumes that the hazard of infection is constant in time,
    not periodic as it is with a periodic birth pulse.'''
    # TODO: Remove `fake`
    def __init__(self, parameters, seed=None, fake=True, debug=False):
        super().__init__(parameters, seed=seed)
        self.population_size = parameters.population_size
        # Reuse `.immune_status_pdf` and `.age`, in case `.pdf()`
        # or `.rvs()` is called repeatedly
        if fake:
            self.immune_status_conditional_pdf = (
                self._immune_status_conditional_pdf_fake)
        else:
            # Interpolate in `age`
            self.immune_status_pdf = _interpolate.Interpolant(
                _solver.solve(parameters, debug=debug))
        self.age = age.Age(parameters, seed=self.rng)

    # TODO: Remove
    @staticmethod
    def _immune_status_conditional_pdf_fake(ages):
        warnings.warn('Fake `initial_conditions`!')
        pdf = pandas.DataFrame(0., index=ages,
                               columns=immune_status.immune_statuses)
        pdf['infectious'] = 0.05
        pdf['susceptible'] = 1. - pdf.sum(axis='columns')
        return pdf

    def immune_status_conditional_pdf(self, ages):
        '''The probability of being in each immune state vs. `ages`,
        conditioned on being alive.'''
        p = self.immune_status_pdf(ages)
        return p.divide(p.sum(axis='columns'), axis='index')

    def pdf(self, ages):
        return (
            self.immune_status_conditional_pdf(ages)
            .multiply(self.age.pdf(ages), axis='index'))

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
                                                       axis='index'))
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

        def has_infected(ser):
            return ser.isin(immune_status.infected).any()

        initial_conditions = self.rvs[InitialConditions]
        return _rv.sample_until(initial_conditions.rvs,
                                has_infected,
                                max_samples=max_samples)

