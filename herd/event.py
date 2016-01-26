import functools


# Use __lt__ and __eq__ to generate all the other comparisons
@functools.total_ordering
class Event:
    def __lt__(self, other):
        return (self.time < other.time)

    def __eq__(self, other):
        return (self.time == other.time)
