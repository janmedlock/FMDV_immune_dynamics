import numpy
import pandas

from . import _event, _rv

# {(a, b): s} is the annual survival for ages [a, b).
_ANNUAL_SURVIVAL = {
    (0, 1): 0.66,
    (1, 3): 0.79,
    (3, 12): 0.88,
    (12, numpy.inf): 0.66,
}


class RV(_rv.MemorylessWaitingTimeMixin, _rv.RV):
    name = 'mortality'

    # Use `pandas.IntervalIndex()` for convenience
    _log_annual_survival = numpy.log(pandas.Series(
        _ANNUAL_SURVIVAL.values(),
        index=pandas.IntervalIndex.from_tuples(_ANNUAL_SURVIVAL.keys(),
                                               closed='left')
    ))

    def hazard(self, age):
        return numpy.asanyarray(-self._log_annual_survival[age])

    def logsf(self, age):
        '''Logarithm of the survival function.'''
        intervals = self._log_annual_survival.index
        # `t` is the amount time lived in each age interval
        t = numpy.clip(age - intervals.left, 0, intervals.length)
        return (t * self._log_annual_survival).sum()

    def sf(self, age):
        '''Survival function.'''
        return numpy.exp(self.logsf(age))

    def cdf(self, age):
        '''Cumulative distribution function.'''
        return -numpy.expm1(self.logsf(age))

    def logcdf(self, age):
        '''The logarithm of the cumulative distribution function.'''
        return numpy.log1p(-self.sf(age))

    def logpdf(self, age):
        '''Logarithm of the probability density function.'''
        return numpy.log(self.hazard(age)) + self.logsf(age)

    def pdf(self, age):
        '''The probability density function.'''
        return numpy.exp(self.logpdf(age))

    def isf(self, q):
        '''Inverse survival function.'''
        logq = numpy.asarray(numpy.log(q))
        isf = numpy.zeros(numpy.shape(q))
        logq_max = logq_min = 0
        for (interval, value) in self._log_annual_survival.items():
            logq_min += (interval.right - interval.left) * value
            # This piece of logsf is the line segment from
            # (interval.left, logq_max) to (interval.right, logq_min).
            mask = (logq_min < logq) & (logq <= logq_max)
            isf[mask] = interval.left + (logq[mask] - logq_max) / value
            logq_max = logq_min
        return isf

    def ppf(self, q):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q)

    def rvs(self, size=None):
        return self.ppf(self.rng.random(size=size))


class Mortality(_event.ForAllMixin, _event.Event):
    '''A buffalo dying.'''

    waiting_time = RV.name

    def do(self):
        self.buffalo.die()


class _Parameters:
    '''Dummy parameters for the mortality random variable.'''
    def __init__(self):
        pass  # There are no parameters.


def sf(age):
    '''The survival function.'''
    return RV(_Parameters()).sf(age)
