'''Common code for running and plotting with varying population size
and susceptibility of the lost-immunity class.'''

import pathlib

import numpy

import baseline
import common
import h5
import herd
from herd.utility import arange
import susceptibility


store_path = pathlib.Path(__file__).with_suffix('.h5')

population_size_default = herd.Parameters().population_size

population_sizes = numpy.hstack((
    arange(100, 900, 100, endpoint=True),
    arange(1000, 5000, 1000, endpoint=True)
))


def _copy_runs(SAT, lost_immunity_susceptibility, nruns, hdfstore_out):
    '''Copy the data from 'susceptibility.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'{lost_immunity_susceptibility=}',
                        f'run<{nruns}'))
    with h5.HDFStore(susceptibility.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(
                chunk, 2,
                population_size=population_size_default
            )
            hdfstore_out.put(chunk)


def run(SAT, population_size, lost_immunity_susceptibility, nruns, hdfstore,
        *args, **kwargs):
    if population_size == population_size_default:
        _copy_runs(SAT, lost_immunity_susceptibility, nruns, hdfstore)
    else:
        parameters_kwds = dict(
            SAT=SAT,
            population_size=population_size,
            lost_immunity_susceptibility=lost_immunity_susceptibility
        )
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(dfr, **parameters_kwds)
            hdfstore.put(dfr)
