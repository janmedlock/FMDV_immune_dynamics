'''Common code for running and plotting with varying population size.'''

import pathlib

import numpy

import herd
import herd.utility

var = 'population_size'

label = 'Population size'

log = True

values = numpy.hstack((
    herd.utility.arange(100, 900, 100, endpoint=True),
    herd.utility.arange(1000, 5000, 1000, endpoint=True),
))

store_path = pathlib.Path(__file__).with_suffix('.h5')


def default(SAT):
    return getattr(herd.Parameters(SAT=SAT), var)
