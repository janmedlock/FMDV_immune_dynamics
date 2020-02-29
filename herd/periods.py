import numpy


# Periodic parameters should append their periods.
periods = []


def get_period():
    # The least common multiple of all the periods.
    return numpy.lcm.reduce(periods)
