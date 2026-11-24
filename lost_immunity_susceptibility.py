'''Common code for running and plotting with varying susceptibility of
lost-immunity true and recovered false.'''

import pathlib

import numpy

import herd

var = 'lost_immunity_susceptibility'

label = 'Susceptibility of lost-immunity true and recovered false'

log = False

values = numpy.linspace(0, 1, 11)

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')
