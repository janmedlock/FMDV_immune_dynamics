'''Common code for running and plotting with varying susceptibility of
the lost-immunity class.'''

import pathlib

import numpy

import baseline
import common
import h5
import herd


store_path = pathlib.Path(__file__).with_suffix('.h5')

susceptibility_default = herd.Parameters().lost_immunity_susceptibility

susceptibilities = numpy.linspace(0, 1, 11)


def _copy_runs(SAT, nruns, hdfstore_out):
    '''Copy the data from 'baseline.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'run<{nruns}'))
    with h5.HDFStore(baseline.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(
                chunk, 2,
                lost_immunity_susceptibility=susceptibility_default
            )
            hdfstore_out.put(chunk)


def run(SAT, lost_immunity_susceptibility, nruns, hdfstore,
        *args, **kwargs):
    if lost_immunity_susceptibility == susceptibility_default:
        _copy_runs(SAT, nruns, hdfstore)
    else:
        parameters_kwds = dict(
            SAT=SAT,
            lost_immunity_susceptibility=lost_immunity_susceptibility,
        )
        parameters = herd.Parameters(**parameters_kwds)
        logging_prefix = common.get_logging_prefix(**parameters_kwds)
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(dfr, **parameters_kwds)
            hdfstore.put(dfr)
