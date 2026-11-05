from . import _rv

sexes = ('female', 'male')


class ToSex(_rv.ChoicesRV):
    '''Random variable for sex of an offspring.'''
    def choices(self, parameters):
        p_female = parameters.female_probability
        return {'female': p_female,
                'male'  : 1 - p_female}


class BuffaloMixin:
    '''Mixin for `..buffalo.Buffalo()`.'''
    @property
    def to_sex(self):
        '''Random variable to determine the sex of a new buffalo.'''
        return self.herd.rvs[ToSex]
