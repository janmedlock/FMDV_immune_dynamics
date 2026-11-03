import numpy

from . import _rv


class RandomVariables:
    '''The random variables for the simulation.'''

    def __init__(self, parameters, seed=None):
        rng = numpy.random.default_rng(seed)
        for rv in _rv.rvs:
            assert not hasattr(self, rv.name)
            setattr(self, rv.name, rv(parameters, seed=rng))
