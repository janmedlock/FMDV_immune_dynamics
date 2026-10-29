import abc
import inspect

import numpy

from .parameters import Parameters

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

    def rvs(self, *args, **kwds):
        return self._rv.rvs(*args, random_state=self.rng, **kwds)

    def __getattr__(self, name):
        '''Try to get the attribute from the `scipy.stats` random variable.'''
        try:
            return getattr(self._rv, name)
        except AttributeError:
            raise AttributeError from None

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))


class RandomVariables:
    '''The random variables for the simulation.'''

    def __init__(self, parameters=None, seed=None):
        rng = numpy.random.default_rng(seed)
        if parameters is None:
            parameters = Parameters(seed=rng)
        for rv in _rvs:
            setattr(self, rv.name, rv(parameters, seed=rng))
