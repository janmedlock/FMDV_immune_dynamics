import functools

import numpy


def _set_random_state(rvs, random_state):
    @functools.wraps(rvs)
    def wrapped(*args, **kwds):
        return rvs(*args, random_state=random_state, **kwds)
    return wrapped


class RV:
    '''A generic random variable.'''
    def __init__(self, seed=None):
        self.rng = numpy.random.default_rng(seed)

    def _copyattrs(self, obj):
        for name in dir(obj):
            if not name.startswith('__') and not hasattr(self, name):
                val = getattr(obj, name)
                if name == 'rvs':
                    val = _set_random_state(val, self.rng)
                setattr(self, name, val)

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))

    def __repr__(self, params=()):
        clsname = f'{self.__module__}.{self.__class__.__name__}'
        if len(params) == 0:
            return f'<{clsname}>'
        paramstr = ', '.join(f'{p} = {getattr(self, p)}' for p in params)
        return f'<{clsname}: {paramstr}>'
