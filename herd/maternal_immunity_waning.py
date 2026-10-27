import scipy

from . import event, rv


class RV(rv.RV):
    '''Time to waning of maternal immunity is gamma distributed
    with rate maternal_immunity_duration_mean and shape
    maternal_immunity_duration_shape.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.maternal_immunity_duration_mean = (
            parameters.maternal_immunity_duration_mean)
        self.maternal_immunity_duration_shape = (
            parameters.maternal_immunity_duration_shape)
        scale = (self.maternal_immunity_duration_mean
                 / self.maternal_immunity_duration_shape)
        dist = scipy.stats.gamma(self.maternal_immunity_duration_shape,
                                 scale=scale)
        super()._copyattrs(dist)

    def __repr__(self):
        return super().__repr__(('maternal_immunity_duration_mean',
                                 'maternal_immunity_duration_shape'))


class MaternalImmunityWaning(event.ForImmuneStatus, event.AgeDependent,
                             immune_status='maternal_immunity'):
    '''A buffalo losing maternal immunity.'''
    def do(self):
        self.buffalo.change_immune_status_to('susceptible')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.maternal_immunity_waning,
            initializing_herd)
