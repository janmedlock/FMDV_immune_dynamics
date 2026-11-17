import abc
import collections
import functools
import inspect

import numpy
import scipy.stats.sampling

# Non-abstract subclasses of `RV()` are added to this set when they
# are defined and then automatically attached to
# `random_variables.RandomVariables()`
rvs = set()


def expon_scale_from_rate(rate):
    '''Get the exponential scale from `rate`.'''
    assert rate >= 0
    return 1 / rate if rate > 0 else numpy.inf


def gamma_scale_from_mean(mean, shape):
    '''Get the gamma scale from `mean` and `shape`.'''
    return mean / shape


def sample_until(rvs_, target, *args, max_samples=1_000_000, **kwds):
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
    @classmethod
    def __init_subclass__(cls, abstract=False, **kwds):
        '''Add non-abstract `cls` to `rv_subclasses`.'''
        super().__init_subclass__(**kwds)
        if inspect.isabstract(cls):
            assert abstract
        else:
            assert cls not in rvs
            rvs.add(cls)

    def __init__(self, parameters, seed=None):
        self.rng = numpy.random.default_rng(seed)


class ChoicesRV(RV,
                abstract=True,
                metaclass=abc.ABCMeta):
    '''A random variable for choices.'''
    @abc.abstractmethod
    def choices(self, parameters):
        '''Return a dictionary with the choices as the keys and their
        probabilities as the values. Subclasses must define this
        method.'''

    def __init__(self, parameters, seed=None):
        super().__init__(parameters, seed=seed)
        choices = self.choices(parameters)
        self.choices_ = numpy.array(tuple(choices.keys()))
        self.p = numpy.array(tuple(choices.values()))

    def rvs(self, **kwds):
        return self.rng.choice(self.choices_, p=self.p, **kwds)


class ScipyRV(RV,
              abstract=True,
              metaclass=abc.ABCMeta):
    '''A random variable based on a random variable from `scipy.stats`.'''
    @abc.abstractmethod
    def rv(self, parameters):
        '''The `scipy.stats` random variable. Subclasses must define
        this method.'''

    def __init__(self, parameters, seed=None):
        super().__init__(parameters, seed=seed)
        self._rv = self.rv(parameters)
        self._rv.random_state = self.rng

    def __getattr__(self, attr):
        '''Try to get the attribute from the underlying `scipy.stats`
        random variable.'''
        try:
            return getattr(self._rv, attr)
        except AttributeError:
            pass
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'")

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))


class WaitingTimeMixin(metaclass=abc.ABCMeta):
    '''A mixin for a waiting-time `RV()` with the
    `time_remaining_rvs()` sampler.'''
    @abc.abstractmethod
    def time_remaining_rvs(self, age, *args, **kwds):
        '''Get a sample time remaining. Subclasses must define this
        method.'''


class MemorylessWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for memoryless waiting times with the
    `.time_remaining_rvs()` sampler.'''
    def time_remaining_rvs(self, age, *args, **kwds):
        '''A memoryless random variable is its own time-remaining
        random variable.'''
        return self.rvs(*args, **kwds)


class AgeDependentWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for age-dependent waiting times with the
    `.time_remaining_rvs()` sampler.'''
    @_sample_nonnegative
    def time_remaining_rvs(self, age, *args, **kwds):
        '''Get a sample time remaining.'''
        # The result might here be negative,
        # but the `_sample_nonnegative()` decorator
        # resamples to get a nonnegative result
        return self.rvs(*args, **kwds) - age


class TimeSinceEntryDependentWaitingTimeMixin(WaitingTimeMixin):
    '''A mixin for time-since-entry-dependent waiting times with the
    `.time_remaining_rvs()` sampler.'''
    # An object with a `.logpdf` attribute for
    # `scipy.stats.sampling.NumericalInversePolynomial()`
    # in `.time_since_entry_rvs()`
    _TimeSinceEntryDist = collections.namedtuple('TimeSinceEntryDist',
                                                 ('logpdf',))

    @functools.cached_property  # Build on first use
    def time_since_entry_rvs(self):
        '''Random sampler for time since entry.'''
        # At equilibrium, the PDF of times since entry given the event
        # has not yet occurred is S(t) / mu, where S(t) is the
        # survival of the waiting time and mu is the mean waiting
        # time. That is, it's proportional to S(t).
        # The normalizing constant is unnecessary
        time_since_entry_logpdf = self.logsf
        time_since_entry = scipy.stats.sampling.NumericalInversePolynomial(
            self._TimeSinceEntryDist(logpdf=time_since_entry_logpdf),
            domain=self.support(), random_state=self.rng)
        return time_since_entry.rvs

    @_sample_nonnegative
    def time_remaining_rvs(self, age, *args, **kwds):
        '''Get a sample time remaining.'''
        # The result might here be negative,
        # but the `_sample_nonnegative()` decorator
        # resamples to get a nonnegative result
        return (self.rvs(*args, **kwds)
                - self.time_since_entry_rvs(*args, **kwds))
