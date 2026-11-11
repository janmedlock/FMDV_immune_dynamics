import numpy
import pandas
import scipy

from .. import _event, _rv

_ANNUAL_SURVIVAL = {
    # `(a, b): s` is the annual survival for ages [a, b)
    (0, 1): 0.66,
    (1, 3): 0.79,
    (3, 12): 0.88,
    (12, numpy.inf): 0.66,
}


def _mortality_rate(annual_survival):
    return {
        ages: -numpy.log(val)
        for (ages, val) in annual_survival.items()
    }


def _crude_mortality_rate(annual_survival):

    def log_one_minus_exp(x):
        '''log(1 - exp(x))'''
        return scipy.special.logsumexp(
            numpy.broadcast_arrays(0, x),
            b=((1,), (-1,)), axis=0
        )

    annual_survival_ = numpy.fromiter(annual_survival.values(), float,
                                      count=len(annual_survival))
    age_interval_length = numpy.fromiter(
        (r - l for (l, r) in annual_survival),
        float, count=len(annual_survival))
    log_conditional_survival_to_end = scipy.special.xlogy(age_interval_length,
                                                          annual_survival_)
    log_survival_to_start = numpy.concat(
        [[0], log_conditional_survival_to_end[:-1].cumsum()]
    )
    with numpy.errstate(divide='ignore'):
        mortality_rate = -numpy.log(annual_survival_)
    mortality_rate_zero = mortality_rate == 0
    log_conditional_mean_survival = numpy.empty_like(annual_survival_)
    log_conditional_mean_survival[~mortality_rate_zero] = (
        log_one_minus_exp(
            log_conditional_survival_to_end[~mortality_rate_zero]
        )
        - numpy.log(mortality_rate[~mortality_rate_zero])
    )
    log_conditional_mean_survival[mortality_rate_zero] = numpy.log(
        age_interval_length[mortality_rate_zero]
    )
    log_mean_age_at_death = scipy.special.logsumexp(
        log_survival_to_start
        + log_conditional_mean_survival
    )
    # Formatted like `_ANNUAL_SURVIVAL`
    return {
        (0, numpy.inf): numpy.exp(-log_mean_age_at_death),  # per year
    }


def _series_with_interval_index(d, closed='left'):
    '''Convert to a `pandas.Series()` with `pandas.IntervalIndex()`.'''
    return pandas.Series(
        d.values(),
        index=pandas.IntervalIndex.from_tuples(d, closed=closed)
    ).sort_index()


class WaitingTime(_rv.AgeDependentWaitingTimeMixin,
                  _rv.RV):

    def __init__(self, parameters, seed=None):
        super().__init__(parameters, seed=seed)
        self.age_dependent = parameters.mortality_age_dependent
        if self.age_dependent:
            hazard = _mortality_rate
        else:
            hazard = _crude_mortality_rate
        self._hazard = _series_with_interval_index(hazard(_ANNUAL_SURVIVAL))
        self.logsf = numpy.vectorize(self._logsf_single, otypes='d')

    def hazard(self, age):
        return numpy.asanyarray(self._hazard[age])

    def _logsf_single(self, age):
        '''Logarithm of the survival function.'''
        intervals = self._hazard.index
        exposure = numpy.clip(age - intervals.left, 0, intervals.length)
        return (-self._hazard * exposure).sum()

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
        for (interval, haz) in self._hazard.items():
            logq_min += -(interval.right - interval.left) * haz
            # This piece of logsf is the line segment from
            # (interval.left, logq_max) to (interval.right, logq_min).
            mask = (logq_min < logq) & (logq <= logq_max)
            isf[mask] = interval.left + (logq_max - logq[mask]) / haz
            logq_max = logq_min
        return isf

    def ppf(self, q):
        '''Percent point function, the inverse of the CDF.'''
        return self.isf(1 - q)

    def rvs(self, size=None):
        return self.isf(self.rng.random(size=size))


class BuffaloMixin:
    '''Mixin for `..buffalo.Buffalo()` to add the method needed by
    `Mortality()`.'''
    def die(self):
        self.herd.remove(self)
        self.events.clear()


class Mortality(_event.ForAllMixin,
                _event.Event):
    '''A buffalo dying.'''
    waiting_time = WaitingTime

    def do(self):
        self.buffalo.die()


class _Parameters:
    '''Dummy parameters for the mortality random variable.'''
    def __init__(self, mortality_age_dependent):
        self.mortality_age_dependent = mortality_age_dependent


def sf(mortality_age_dependent, age):
    '''The survival function.'''
    params = _Parameters(mortality_age_dependent)
    waiting_time = WaitingTime(params)
    return waiting_time.sf(age)
