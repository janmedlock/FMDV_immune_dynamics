# Use triangular birth hazard.
from .birth_triangular import *

from . import event
from . import buffalo


class Event(event.Event):
    def __init__(self, buff):
        self.buffalo = buff

        assert self.buffalo.sex == 'female'

        self.set_next_birth_time()

    def set_next_birth_time(self):
        self.time = (self.buffalo.herd.time
                      + self.buffalo.herd.rvs.birth.rvs(self.buffalo.herd.time,
                                                        self.buffalo.age()))

    def __call__(self):
        if self.buffalo.immune_status == 'recovered':
            calf_status = 'maternal immunity'
        else:
            calf_status = 'susceptible'

        self.buffalo.herd.append(buffalo.Buffalo(self.buffalo.herd,
                                                 calf_status))

        self.set_next_birth_time()
