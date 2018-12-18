from numpy import exp


class RV:
    '''A generic random variable.'''
    def _copyattrs(self, obj):
        for x in dir(obj):
            if not x.startswith('__') and not hasattr(self, x):
                setattr(self, x, getattr(obj, x))

    def hazard(self, age):
        return exp(self.logpdf(age) - self.logsf(age))

    def __repr__(self, params = ()):
        cls = self.__class__
        clsname = '{}.{}'.format(self.__module__, self.__class__.__name__)
        paramstrs = ['{} = {}'.format(p, getattr(self, p))
                     for p in params]
        if len(params) == 0:
            return '<{}>'.format(clsname)
        else:
            return '<{}: {}>'.format(clsname, ', '.join(paramstrs))
