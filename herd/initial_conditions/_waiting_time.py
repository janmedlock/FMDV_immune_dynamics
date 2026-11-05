'''Handle waiting times.'''


class WaitingTimes:
    '''Helper to get parameterized instances of waiting times.'''
    def __init__(self, parameters):
        self.parameters = parameters
        self._d = {}  # Cache

    def __getitem__(self, waiting_time):
        try:
            return self._d[waiting_time]
        except KeyError:
            pass
        val = self._d[waiting_time] = waiting_time(self.parameters)
        return val


def name(waiting_time):
    module = waiting_time.__module__
    return module.split('.')[-1]
