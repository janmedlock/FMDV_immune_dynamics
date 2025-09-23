'''Common code for running and plotting with varying population size.'''

import pathlib

import numpy

import herd
import herd.utility


var = 'population_size'

label = 'Population\nsize'

log = True

values = numpy.hstack((
    herd.utility.arange(100, 900, 100, endpoint=True),
    herd.utility.arange(1000, 5000, 1000, endpoint=True),
))

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')
