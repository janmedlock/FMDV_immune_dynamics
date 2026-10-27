import scipy

from . import event, rv


class RV(rv.RV):
    '''Gamma-distributed progression period with rate
    progression_mean and shape progression_shape.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.progression_mean = parameters.progression_mean
        self.progression_shape = parameters.progression_shape
        scale = self.progression_mean / self.progression_shape
        distn = scipy.stats.gamma(self.progression_shape, scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('progression_mean', 'progression_shape'))


class Progression(event.ForImmuneStatus, event.TimeSinceEntryDependent,
                  immune_status='exposed'):
    '''A buffalo becoming infectious.'''
    def do(self):
        self.buffalo.change_immune_status_to('infectious')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.progression,
            initializing_herd)
