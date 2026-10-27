import numpy
import scipy

from . import event, rv


class RV(rv.RV):
    '''Waiting time to loss of antibodies is exponential distributed.'''
    def __init__(self, parameters, seed=None):
        super().__init__(seed=seed)
        self.antibody_loss_hazard = parameters.antibody_loss_hazard
        assert self.antibody_loss_hazard >= 0
        if self.antibody_loss_hazard > 0:
            scale = 1 / self.antibody_loss_hazard
        else:  # self.antibody_loss_hazard = 0
            scale = numpy.inf
        distn = scipy.stats.expon(scale=scale)
        super()._copyattrs(distn)

    def __repr__(self):
        return super().__repr__(('antibody_loss_hazard', ))


class AntibodyLoss(event.ForImmuneStatuses, event.Memoryless,
                   immune_statuses={'recovered_true', 'recovered_false'}):
    '''A buffalo losing apparent antibodies.'''
    def do(self):
        if self.buffalo.herd.rvs.antibody_loss_to_lost_immunity_true.rvs():
            self.buffalo.change_immune_status_to('lost_immunity_true')
        else:
            self.buffalo.change_immune_status_to('lost_immunity_false')

    def sample_waiting_time(self, initializing_herd=False):
        return self.sample_waiting_time_from(
            self.buffalo.herd.rvs.antibody_loss)
