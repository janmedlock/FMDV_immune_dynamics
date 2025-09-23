'''Common code to run and plot one simulation for each SAT.'''

import pathlib

import baseline
import common
import herd


SEED = 0xa70a_8239

store_path = pathlib.Path(__file__).with_suffix('.h5')


def _save_result(store, result):
    '''Save the raw simulation data.'''
    common.save_result(store, result,
                       simulation=True)


def run(SAT, store, seed=None):
    '''Run one simulation.'''
    if seed is None:
        seed = SEED
    p = herd.Parameters(SAT=SAT)
    result = baseline.run_one(p, seed)
    common.prepend_index_levels(result, SAT=SAT)
    _save_result(store, result)
