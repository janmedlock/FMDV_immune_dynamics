'''Handle random variables.'''

from .. import events


class WaitingTimes:
    '''Helper to get parameterized instances of waiting times.'''
    def __init__(self, parameters):
        self.parameters = parameters
        self._d = {}  # Cache

    def __getitem__(self, key):
        try:
            return self._d[key]
        except KeyError:
            pass
        module = getattr(events, key)
        val = self._d[key] = module.WaitingTime(self.parameters)
        return val
