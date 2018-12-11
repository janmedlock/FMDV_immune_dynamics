import abc
from functools import total_ordering


# Use __lt__ and __eq__ to generate all the other comparisons
@total_ordering
class Event(abc.ABC):
    @abc.abstractmethod
    def is_valid(self):
        '''Check whether the buffalo is valid to have the event.
        Subclasses must define this method.'''

    @abc.abstractmethod
    def do(self):
        '''Execute the event.
        Subclasses must define this method.'''

    @abc.abstractmethod
    def sample_time(self):
        '''Generate a random sample time for the event.
        Subclasses must define this method.'''

    @property
    def name(self):
        return self.__class__.__name__

    def __init__(self, buffalo):
        self.buffalo = buffalo
        assert self.is_valid()
        self.time = self.sample_time()

    def __call__(self):
        assert self.is_valid()
        self.do()

    def __lt__(self, other):
        return (self.time < other.time)

    def __eq__(self, other):
        return (self.time == other.time)

    def __repr__(self):
        return 't = {}: {} for buffalo #{}'.format(
            self.time, self.name, self.buffalo.identifier)


class Events(dict):
    # Consider storing the events in a data type that's more
    # efficient to find the minimum time.
    def __init__(self, buffalo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffalo = buffalo

    def add(self, event):
        self[event.name] = event

    def remove(self, event):
        del self[event.name]

    def update(self, event):
        self.add(event)

    def get_next(self):
        return min(self.values())
