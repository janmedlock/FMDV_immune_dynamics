from . import rv
from . import event


class gen(rv.deterministic):
    'Transition at maternal_immunity_duration with probability 1.'

    def __init__(self, parameters, *args, **kwargs):
        super().__init__('maternal_immunity_duration',
                         parameters.maternal_immunity_duration,
                         *args, **kwargs)


class Event(event.Event):
    def __init__(self, buffalo):
        self.buffalo = buffalo

        self.time = (self.buffalo.birth_date
                     + self.buffalo.herd.rvs.maternal_immunity_waning.rvs())

        assert self.time >= self.buffalo.herd.time

    def __call__(self):
        assert self.buffalo.immune_status == 'maternal immunity'

        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].remove(
            self.buffalo)
        self.buffalo.immune_status = 'susceptible'
        self.buffalo.herd.by_immune_status[self.buffalo.immune_status].append(
            self.buffalo)

        self.buffalo.update_infection_time()

        try:
            del self.buffalo.events['maternal immunity waning']
        except KeyError:
            pass
