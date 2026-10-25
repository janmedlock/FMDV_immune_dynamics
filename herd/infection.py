import numpy


class gen:
    '''Time to infection is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        self.rng = numpy.random.default_rng(seed)

    def rvs(self, rate, size=None):
        scale = 1 / rate if rate > 0 else numpy.inf
        return self.rng.exponential(scale)
