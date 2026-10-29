import abc
import collections
import functools
import inspect

import numpy
import scipy.stats.sampling

# Non-abstract subclasses of `RV()` are added to this list when they
# are defined
_rvs = []


def expon_scale_from_rate(rate):
    '''Get the exponential scale from `rate`.'''
    assert rate >= 0
    return 1 / rate if rate > 0 else numpy.inf


def gamma_scale_from_mean(mean, shape):
    '''Get the gamma scale from `mean` and `shape`.'''
    return mean / shape


class RV(metaclass=abc.ABCMeta):
    '''A generic random variable.'''
    @property
    @abc.abstractmethod
    def name(self):
        '''Name of the random variable for `.RandomVariables()`'''

    @classmethod
    def __init_subclass__(cls, **kwds):
        '''Add non-abstract `cls` to `_rvs`.'''
        super().__init_subclass__(**kwds)
        if not inspect.isabstract(cls):
            _rvs.append(cls)

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


class TimeSinceEntryMixin:
    '''A mixin for waiting times to build `.time_since_entry`,
    used by `.event.TimeSinceEntryDependentEvent()`.'''

    # An object with a `.logpdf` attribute for
    # `scipy.stats.sampling.NumericalInversePolynomial()`
    # in `._build_time_since_entry_rv()`
    _TimeSinceEntryDist = collections.namedtuple('TimeSinceEntryDist',
                                                 ('logpdf',))

    @functools.cached_property  # Build on first use
    def time_since_entry(self):
        r'''Build a random variable for time since entry from a
        waiting time. At equilibrium, the PDF of times since entry
        given the event has not yet occurred is S(t) / mu, where S(t)
        is the survival of the waiting time and mu is the mean waiting
        time. That is, it's proportional to S(t).'''
        # The normalizing constant is unnecessary
        time_since_entry_logpdf = self.logsf
        return scipy.stats.sampling.NumericalInversePolynomial(
            self._TimeSinceEntryDist(logpdf=time_since_entry_logpdf),
            domain=self.support(), random_state=self.rng)


class RandomVariables:
    '''The random variables for the simulation.'''

    def __init__(self, parameters, seed=None):
        rng = numpy.random.default_rng(seed)
        for rv in _rvs:
            assert not hasattr(self, rv.name)
            setattr(self, rv.name, rv(parameters, seed=rng))
