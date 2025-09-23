'''Common code for running and plotting with varying both population
size and susceptibility of the lost-immunity class.'''

import pathlib

import baseline
import common
import h5
import herd
import population_size as population_size_
import susceptibility


store_path = pathlib.Path(__file__).with_suffix('.h5')


def _save_result(store, result):
    '''Save extinction time.'''
    # If you change this, you must change `_copy_population_size()`
    # and `_copy_susceptibility()` to save matching output.
    common.save_result(store, result,
                       extinction_time=True)


def _copy_population_size(store, nruns, SAT,
                          population_size, **kwds):
    '''Copy the data from 'population_size.h5'.'''
    extinction_time = h5.load(population_size_.store_path, 'extinction_time')
    index = extinction_time.index.to_frame()
    mask = (
        (index['SAT'] == SAT)
        & (index['population_size'] == population_size)
        & (index['run'] < nruns)
    )
    extinction_time_masked = extinction_time[mask]
    common.insert_index_levels(extinction_time_masked, 2, **kwds)
    store.put('extinction_time', extinction_time_masked)


def _copy_susceptibility(store, nruns, SAT,
                         lost_immunity_susceptibility, **kwds):
    '''Copy the data from 'susceptibility.h5'.'''
    extinction_time = h5.load(susceptibility.store_path, 'extinction_time')
    index = extinction_time.index.to_frame()
    mask = (
        (index['SAT'] == SAT)
        & (index['lost_immunity_susceptibility']
           == lost_immunity_susceptibility)
        & (index['run'] < nruns)
    )
    extinction_time_masked = extinction_time[mask]
    common.insert_index_levels(extinction_time_masked, 3, **kwds)
    store.put('extinction_time', extinction_time_masked)


def _is_default(module, val):
    return (module.default == val)


def run(SAT, lost_immunity_susceptibility, population_size, nruns, store,
        copy_only, *args, **kwargs):
    parameters_kwds = {
        'SAT': SAT,
        'lost_immunity_susceptibility': lost_immunity_susceptibility,
        'population_size': population_size,
    }
    if _is_default(susceptibility, lost_immunity_susceptibility):
        _copy_population_size(store, nruns, **parameters_kwds)
        stored = True
    elif _is_default(population_size_, population_size):
        _copy_susceptibility(store, nruns, **parameters_kwds)
        stored = True
    elif not copy_only:
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for chunk in chunks:
            common.prepend_index_levels(chunk, **parameters_kwds)
            _save_result(store, chunk)
        stored = True
    else:
        stored = False
    return stored
