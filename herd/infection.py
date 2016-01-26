from scipy import stats

from . import event
from . import recovery


class Event(event.Event):
    ## Fix me! ##
    def __init__(self, buffalo):
        self.buffalo = buffalo

        assert self.buffalo.is_susceptible()

        if (self.buffalo.herd.number_infectious > 0):
            force_of_infection = (self.buffalo.herd.rvs.transmission_rate
                                  * self.buffalo.herd.number_infectious)

            infection_time = stats.expon.rvs(
                scale = 1 / force_of_infection)

            self.time = self.buffalo.herd.time + infection_time
        else:
            self.time = None

    def __call__(self):
        assert self.buffalo.is_susceptible()

        self.buffalo.change_immune_status_to('infectious')

        try:
            del self.buffalo.events['infection']
        except KeyError:
            pass
        
        self.buffalo.events['recovery'] = recovery.Event(self.buffalo)
