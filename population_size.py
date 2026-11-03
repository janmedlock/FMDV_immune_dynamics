'''Common code for running and plotting with varying population size.'''

import pathlib

import numpy

import common
import herd

var = 'population_size'

label = 'Population size'

log = True

values = numpy.hstack((
    common.arange(100, 900, 100, endpoint=True),
    common.arange(1000, 5000, 1000, endpoint=True),
))

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')
