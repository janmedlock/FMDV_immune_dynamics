'''Common code for running and plotting with varying susceptibility of
the lost-immunity class.'''

import pathlib

import numpy

import baseline
import common
import h5
import herd


var = 'lost_immunity_susceptibility'

label = 'Susceptibility\nof lost-immunity\nstate'

log = False

values = numpy.linspace(0, 1, 11)

default = getattr(herd.Parameters(), var)

store_path = pathlib.Path(__file__).with_suffix('.h5')


def _copy_runs(hdfstore_out, nruns, SAT, **kwds):
    '''Copy the data from 'baseline.h5'.'''
    where = ' & '.join((f'{SAT=}',
                        f'run<{nruns}'))
    with h5.HDFStore(baseline.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(chunk, 2, **kwds)
            hdfstore_out.put(chunk)


def run(SAT, lost_immunity_susceptibility, nruns, hdfstore,
        *args, **kwargs):
    parameters_kwds = dict(
        SAT=SAT,
        lost_immunity_susceptibility=lost_immunity_susceptibility,
    )
    if lost_immunity_susceptibility == default:
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
