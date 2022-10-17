'''Common code for running and plotting with varying population size.'''

import pathlib

import numpy

import baseline
import common
import h5
import herd
from herd.utility import arange


store_path = pathlib.Path(__file__).with_suffix('.h5')

default = herd.Parameters().population_size

population_sizes = numpy.hstack((
    arange(100, 900, 100, endpoint=True),
    arange(1000, 5000, 1000, endpoint=True)
))


def _copy_runs(hdfstore_out, nruns, SAT, **kwds):
    '''Copy the data from 'baseline.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'run<{nruns}'))
    with h5.HDFStore(baseline.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(chunk, 2, **kwds)
            hdfstore_out.put(chunk)


def run(SAT, population_size, nruns, hdfstore,
        *args, **kwargs):
    parameters_kwds = dict(
        SAT=SAT,
        population_size=population_size,
    )
    if population_size == default:
        _copy_runs(hdfstore, nruns, **parameters_kwds)
    else:
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(dfr, **parameters_kwds)
            hdfstore.put(dfr)
