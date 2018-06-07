import numpy
import pandas

from . import rv


class gen(rv.RV):
    # {(a, b): s} is the annual survival for ages [a, b).
    _annual_survival = {(0, 1): 0.66,
                        (1, 3): 0.79,
                        (3, 12): 0.88,
                        (12, numpy.inf): 0.66}
    # Convert to pandas.Interval()'s for convenience.
    _annual_survival = pandas.Series(
        {pandas.Interval(*interval, closed='left'): value
         for (interval, value) in _annual_survival.items()}).sort_index()

    def __init__(self, parameters):
        pass

    def hazard(self, age):
        return -numpy.log(self._annual_survival[age])

    def logsf(self, age):
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

    def sf(self, age):
        '''Survival function.'''
        return numpy.exp(self.logsf(age))

    def isf(self, q):
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

    def cdf(self, age):
        '''Cumulative distribution function.'''
        return 1 - self.sf(age)

    def ppf(self, q):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q)

    def logpdf(self, age):
        '''Logarithm of the probability density function.'''
        return numpy.log(self.hazard(age)) + self.logsf(age)

    def pdf(self, age):
        '''Probability density function.'''
        return numpy.exp(self.logpdf(age))

    def rvs(self, size=None):
        U = numpy.random.random_sample(size=size)
        return self.ppf(U)
