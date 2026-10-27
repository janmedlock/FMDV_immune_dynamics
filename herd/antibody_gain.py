import numpy
import scipy

from . import event, rv


class RV(rv.RV):
    '''Waiting time to gain of antibodies is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_gain_hazard = parameters.antibody_gain_hazard
        assert self.antibody_gain_hazard >= 0
        if self.antibody_gain_hazard > 0:
            scale = 1 / self.antibody_gain_hazard
        else:  # self.antibody_gain_hazard = 0
            scale = numpy.inf
        distn = scipy.stats.expon(scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('antibody_gain_hazard', ))


class AntibodyGain(event.ForImmuneStatuses, event.Memoryless,
                   immune_statuses={'lost_immunity_true',
                                    'lost_immunity_false'}):
    '''A buffalo gaining apparent antibodies.'''
    def do(self):
        if self.buffalo.herd.rvs.antibody_gain_to_recovered_true.rvs():
            self.buffalo.change_immune_status_to('recovered_true')
        else:
            self.buffalo.change_immune_status_to('recovered_false')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.antibody_gain)
