'''Common code for running and plotting with varying probability of
lost-immunity true.'''

import pathlib

import numpy

import herd

var = 'lost_immunity_true_probability'

label = 'Probability of entering lost-immunity true'

log = False

values = numpy.linspace(0, 1, 11)

store_path = pathlib.Path(__file__).with_suffix('.h5')


def default(SAT):
    return getattr(herd.Parameters(SAT=SAT), var)
