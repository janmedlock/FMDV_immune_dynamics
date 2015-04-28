import numpy
from scipy import stats


class RV(object):
    def _copyattrs(self, obj):
        for x in dir(obj):
            if not hasattr(self, x) and not x.startswith('__'):
                setattr(self, x, getattr(obj, x))

    def __repr__(self, params = ()):
        cls = self.__class__
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)

        paramstrs = ['{} = {}'.format(p, getattr(self, p))
                     for p in params]

        if len(params) == 0:
            return '<{}>'.format(clsname)
        else:
            return '<{}: {}>'.format(clsname, ', '.join(paramstrs))


class deterministic(RV, stats.rv_continuous):
    def __init__(self, paramname = '_scale', scale = 1., *args, **kwargs):
        self._scale = scale

        self._paramname = paramname

        stats.rv_continuous.__init__(self, a = scale, b = scale,
                                     *args, **kwargs)

    def _cdf(self, age):
        return numpy.where(age < self._scale, 0., 1.)

    def _ppf(self, age):
        return self._scale * numpy.ones_like(age)
    
    def _rvs(self):
        return self._scale * numpy.ones(self._size)

    def _munp(self, n, *args):
        if n == 1:
            return self._scale
        else:
            return 0.

    def __repr__(self):
        return RV.__repr__(self, (self._paramname, ))
