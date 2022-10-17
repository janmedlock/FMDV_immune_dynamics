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


def _copy_runs_susceptibility(hdfstore_out, nruns, SAT,
                              lost_immunity_susceptibility, **kwds):
    '''Copy the data from 'susceptibility.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'{lost_immunity_susceptibility=}',
                        f'run<{nruns}'))
    with h5.HDFStore(susceptibility.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(chunk, 2, **kwds)
            hdfstore_out.put(chunk)


def _copy_runs_population_size(hdfstore_out, nruns, SAT,
                               population_size, **kwds):
    '''Copy the data from 'population_size.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'{population_size=}',
                        f'run<{nruns}'))
    with h5.HDFStore(population_size_.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(chunk, 3, **kwds)
            hdfstore_out.put(chunk)


def run(SAT, population_size, lost_immunity_susceptibility, nruns, hdfstore,
        *args, **kwargs):
    parameters_kwds = dict(
        SAT=SAT,
        population_size=population_size,
        lost_immunity_susceptibility=lost_immunity_susceptibility
    )
    if population_size == population_size_.default:
        _copy_runs_susceptibility(hdfstore, nruns, **parameters_kwds)
    elif lost_immunity_susceptibility == susceptibility.default:
        _copy_runs_population_size(hdfstore, nruns, **parameters_kwds)
    else:
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(dfr, **parameters_kwds)
            hdfstore.put(dfr)
