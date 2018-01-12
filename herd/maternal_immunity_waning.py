from scipy import stats

from . import rv


class gen(rv.RV):
    '''
    Time to waning of maternal immunity is gamma distributed
    with rate maternal_immunity_duration_mean and shape
    maternal_immunity_duration_shape.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.maternal_immunity_duration_mean \
            = parameters.maternal_immunity_duration_mean
        self.maternal_immunity_duration_shape \
            = parameters.maternal_immunity_duration_shape
        dist = stats.gamma(self.maternal_immunity_duration_shape,
                           scale = (self.maternal_immunity_duration_mean
                                    / self.maternal_immunity_duration_shape))
        super()._copyattrs(dist)

    def __repr__(self):
        return super().__repr__(('maternal_immunity_duration_mean',
                                 'maternal_immunity_duration_shape'))
