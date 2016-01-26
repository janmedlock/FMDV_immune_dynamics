from scipy import stats

from . import rv
from . import event


class gen(rv.RV):
    '''
    Exponential waiting time to recovery with rate
    recovery_infection_duration.
    '''

    def __init__(self, parameters, *args, **kwargs):
        self.infection_duration = parameters.recovery_infection_duration

        distn = stats.expon(scale = self.infection_duration)
        super()._copyattrs(distn)

    def __repr__(self):
        return rv.RV.__repr__(self, ('infection_duration', ))


class Event(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        self.time = (self.buffalo.herd.time
                     + self.buffalo.herd.rvs.recovery.rvs())

    def __call__(self):
        assert self.buffalo.is_infectious()

        self.buffalo.change_immune_status_to('recovered')

        try:
            del self.buffalo.events['recovery']
        except KeyError:
            pass
