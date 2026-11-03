'''Common code to run and plot one simulation for each SAT.'''

import pathlib

import numpy

import baseline
import common
import herd

store_path = pathlib.Path(__file__).with_suffix('.h5')


def _save_result(store, result):
    '''Save the raw simulation data.'''
    common.save_result(store, result,
                       simulation=True)


def run(SAT, store, seed=common.SEED):
    '''Run one simulation.'''
    rng = numpy.random.default_rng(seed)
    p = herd.Parameters(SAT=SAT, seed=rng)
    result = baseline.run_one(p, seed=rng)
    common.prepend_index_levels(result, SAT=SAT)
    _save_result(store, result)
