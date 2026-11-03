import abc
import collections
import functools
import inspect

import numpy
import scipy.stats.sampling

# Non-abstract subclasses of `RV()` are added to this list when they
# are defined and then automatically attached to
# `random_variables.RandomVariables()`
rvs = []


def expon_scale_from_rate(rate):
    '''Get the exponential scale from `rate`.'''
    assert rate >= 0
    return 1 / rate if rate > 0 else numpy.inf


def gamma_scale_from_mean(mean, shape):
    '''Get the gamma scale from `mean` and `shape`.'''
    return mean / shape


def sample_until(rvs_, target, *args, max_samples=10_000, **kwds):
    '''Sample until `target()` is `True`.'''
    for _ in range(max_samples):
        val = rvs_(*args, **kwds)
        if target(val):
            return val
    raise ValueError(f'Sampling failed after {max_samples=}!')


def _sample_nonnegative(rvs_):
    '''Decorator to sample until nonnegative.'''

    def is_nonnegative(val):
        return val >= 0

    @functools.wraps(rvs_)
    def sample_nonnegative(*args, **kwds):
        return sample_until(rvs_, is_nonnegative, *args, **kwds)

    return sample_nonnegative


class RV(metaclass=abc.ABCMeta):
    '''A generic random variable.'''
    @property
    @abc.abstractmethod
    def name(self):
        '''Name of the random variable for
        `.random_variables.RandomVariables()`'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to `rvs`.'''
        super().__init_subclass__(**kwds)
        if not inspect.isabstract(cls):
            rvs.append(cls)

    def __init__(self, parameters, seed=None):
        self.rng = numpy.random.default_rng(seed)


class ScipyRV(RV, metaclass=abc.ABCMeta):
    '''A random variable based on a random variable from `scipy.stats`.'''
    @abc.abstractmethod
    def rv(self, parameters):
        '''Return the `scipy.stats` random variable.'''

    def __init__(self, parameters, seed=None):
        super().__init__(parameters, seed=seed)
        self._rv = self.rv(parameters)
        self._rv.random_state = self.rng

    def __getattr__(self, name):
        '''Try to get the attribute from the `scipy.stats` random variable.'''
        try:
            return getattr(self._rv, name)
        except AttributeError:
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))


class WaitingTimeMixin(metaclass=abc.ABCMeta):
    '''A mixin for waiting times.'''

    @abc.abstractmethod
    def _time_remaining_rvs(self, *args, **kwds):
        '''Get a sample time remaining, used by
        `.time_remaining()`. Subclasses must define this method.'''

    # An object with a `.rvs()` method for `.time_remaining()`
    _TimeRemaining = collections.namedtuple('WaitingTimeRemaining',
                                            ('rvs',))

    @functools.cached_property  # Build on first use
    def time_remaining(self):
        '''A random sampler for waiting time remaining given the event
        has not yet occurred.'''
        return self._TimeRemaining(rvs=self._time_remaining_rvs)


class MemorylessWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for memoryless waiting times to set the
    `.time_remaining` sampler.'''

    def _time_remaining_rvs(self, *args, **kwds):
        '''A memoryless random variable is its own `.time_remaining`.'''
        return self.rvs(*args, **kwds)


class AgeDependentWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for age-dependent waiting times to build the
    `.time_remaining` sampler.'''

    @_sample_nonnegative
    def _time_remaining_rvs(self, *args, **kwds):
        '''Get a sample time remaining.'''
        # The result might here be negative,
        # but the `sample_nonnegative()` decorator
        # resamples to get a nonnegative result
        return self.rvs(*args, **kwds) - self.buffalo.age


class TimeSinceEntryDependentWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for waiting times to build the `.time_remaining`
    sampler.'''

    # An object with a `.logpdf` attribute for
    # `scipy.stats.sampling.NumericalInversePolynomial()`
    # in `._build_time_since_entry_rv()`
    _TimeSinceEntryDist = collections.namedtuple('TimeSinceEntryDist',
                                                 ('logpdf',))

    @functools.cached_property  # Build on first use
    def time_since_entry(self):
        r'''Build a random sampler for time since entry from a
        waiting time. At equilibrium, the PDF of times since entry
        given the event has not yet occurred is S(t) / mu, where S(t)
        is the survival of the waiting time and mu is the mean waiting
        time. That is, it's proportional to S(t).'''
        # The normalizing constant is unnecessary
        time_since_entry_logpdf = self.logsf
        return scipy.stats.sampling.NumericalInversePolynomial(
            self._TimeSinceEntryDist(logpdf=time_since_entry_logpdf),
            domain=self.support(), random_state=self.rng)

    @_sample_nonnegative
    def _time_remaining_rvs(self, *args, **kwds):
        '''Get a sample time remaining.'''
        # The result might here be negative,
        # but the `sample_nonnegative()` decorator
        # resamples to get a nonnegative result
        return (self.rvs(*args, **kwds)
                - self.time_since_entry.rvs(*args, **kwds))
