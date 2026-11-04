import collections.abc

import numpy

from . import _rv


class RandomVariables(collections.abc.Mapping):
    '''The random variables for the simulation.'''

    def __init__(self, parameters, seed=None):
        rng = numpy.random.default_rng(seed)
        self._d = {
            rv_type: rv_type(parameters, seed=rng)
            for rv_type in _rv.rv_types
        }

    def __getitem__(self, key):
        return self._d[key]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __contains__(self, key):
        return key in self._d

    def __eq__(self, other):
        return self._d == other

    def __ne__(self, other):
        return self._d != other

    def keys(self):
        return self._d.keys()

    def items(self):
        return self._d.items()

    def values(self):
        return self._d.values()

    def get(self, key, default=None):
        return self._d.get(key, default)
