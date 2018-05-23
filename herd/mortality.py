import numpy
import pandas
from scipy import integrate, stats

from . import rv


class gen(rv.RV, stats.rv_continuous):
    # {(a, b): s} is the annual survival for ages [a, b).
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.inf): 0.66}
    # Convert to pandas.Interval()'s for convenience.
    _annual_survival = pandas.Series(
        {pandas.Interval(*interval, closed='left'): value
         for (interval, value) in _annual_survival.items()}).sort_index()

    def __init__(self, parameters, *args, **kwargs):
        stats.rv_continuous.__init__(self, name='mortality', a=0, *args, **kwargs)

    def hazard(self, age):
        return -numpy.log(self._annual_survival[age])

    def _logsf(self, age):
        '''Logarithm of the survival function.'''
        logsf = numpy.zeros_like(age)
        for (interval, value) in self._annual_survival.items():
            a = numpy.clip(age, interval.left, interval.right)
            # (a - interval.left) is
            # 0 if age <= interval.left,
            # (interval.right - interval.left) if age >= interval.right,
            # (age - interval.left) otherwise.
            logsf += (a - interval.left) * numpy.log(value)
        return logsf

    def _sf(self, age):
        '''Survival function.'''
        return numpy.exp(self._logsf(age))

    def _isf(self, q):
        '''Inverse survival function.'''
        logq = numpy.asarray(numpy.log(q))
        isf = numpy.zeros_like(q)
        logq_max = logq_min = 0
        for (interval, value) in self._annual_survival.items():
            logq_min += (interval.right - interval.left) * numpy.log(value)
            # This piece of logsf is the line segment from
            # (interval.left, logq_max) to (interval.right, logq_min).
            mask = (logq_min < logq) & (logq <= logq_max)
            isf[mask] = (interval.left
                         + (logq[mask] - logq_max) / numpy.log(value))
            logq_max = logq_min
        return isf

    def _cdf(self, age):
        '''Cumulative distribution function.'''
        return 1 - self._sf(age)

    def _ppf(self, q):
        '''Percent point function, the inverse of the CDF.'''
        return self._isf(1 - q)

    def _logpdf(self, age):
        '''Logarithm of the probability density function.'''
        return numpy.log(self.hazard(age)) + self._logsf(age)

    def _pdf(self, age):
        '''Probability density function.'''
        return numpy.exp(self._logpdf(age))
