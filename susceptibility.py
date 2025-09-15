'''Common code for running and plotting with varying susceptibility of
the lost-immunity class.'''

import pathlib

import numpy

import herd


var = 'lost_immunity_susceptibility'

label = 'Susceptibility\nof lost-immunity\nstate'

log = False

values = numpy.linspace(0, 1, 11)

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')
