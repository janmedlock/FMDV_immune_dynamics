import functools


# Use __lt__ and __eq__ to generate all the other comparisons
@functools.total_ordering
class Event:
    def __init__(self, time, func, label):
        self.time = time
        self.func = func
        self.label = label

    def __lt__(self, other):
        return (self.time < other.time)

    def __eq__(self, other):
        return (self.time == other.time)

    def __call__(self):
        return self.func()
