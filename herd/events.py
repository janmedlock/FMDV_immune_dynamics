from functools import total_ordering


# Use __lt__ and __eq__ to generate all the other comparisons
@total_ordering
class Event:
    def __init__(self, time, fcn, identifier):
        self.time = time
        self.fcn = fcn
        self.identifier = identifier

    def __lt__(self, other):
        return (self.time < other.time)

    def __eq__(self, other):
        return (self.time == other.time)

    def __call__(self, *args, **kwargs):
        self.fcn(*args, **kwargs)

    def __repr__(self):
        return 't = {}: {} for buffalo #{}'.format(
            self.time,
            self.fcn.__name__,
            self.identifier)


class Events(dict):
    def __init__(self, buffalo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffalo = buffalo

    def add(self, name, time):
        self[name] = Event(time,
                           getattr(self.buffalo, name),
                           self.buffalo.identifier)

    def remove(self, name):
        del self[name]

    def update(self, name, time):
        self.add(name, time)

    def get_next(self):
        return min(self.values())
