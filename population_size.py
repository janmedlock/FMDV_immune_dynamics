'''Common code for running and plotting with varying population size.'''

import pathlib

import numpy

import herd
from herd.utility import arange


var = 'population_size'

label = 'Population\nsize'

log = True

values = numpy.hstack((
    arange(100, 900, 100, endpoint=True),
    arange(1000, 5000, 1000, endpoint=True)
))

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')
