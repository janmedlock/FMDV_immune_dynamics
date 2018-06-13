import numpy


class RV:
    def _copyattrs(self, obj):
        for x in dir(obj):
            if not hasattr(self, x) and not x.startswith('__'):
                setattr(self, x, getattr(obj, x))

    def hazard(self, age):
        return numpy.exp(self.logpdf(age) - self.logsf(age))

    def __repr__(self, params = ()):
        cls = self.__class__
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramstrs = ['{} = {}'.format(p, getattr(self, p))
                     for p in params]
        if len(params) == 0:
            return '<{}>'.format(clsname)
        else:
            return '<{}: {}>'.format(clsname, ', '.join(paramstrs))
