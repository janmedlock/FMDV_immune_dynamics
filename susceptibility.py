'''Common code for running and plotting with varying susceptibility of
the lost-immunity class.'''

import pathlib

import numpy

import baseline
import common
import h5
import herd


store_path = pathlib.Path(__file__).with_suffix('.h5')

susceptibilities = numpy.linspace(0, 1, 11)


def _copy_run(SAT, susceptibility, nruns, hdfstore_out):
    '''Copy the data from 'baseline.h5'.'''
    where = f'SAT={SAT} & run<{nruns}'
    with h5.HDFStore(baseline.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            common.insert_index_levels(
                chunk, 2,
                lost_immunity_susceptibility=susceptibility
            )
            hdfstore_out.put(chunk)


def run(SAT, susceptibility, nruns, hdfstore, *args, **kwargs):
    if susceptibility == 1:
        _copy_run(SAT, susceptibility, nruns, hdfstore)
    else:
        parameters = herd.Parameters(
            SAT=SAT,
            lost_immunity_susceptibility=susceptibility
        )
        logging_prefix = ', '.join((
            f'{SAT=}',
            f'lost_immunity_susceptibility={susceptibility}'
        ))
        chunks = baseline.run_many_chunked(parameters, nruns, *args,
                                           logging_prefix=logging_prefix,
                                           **kwargs)
        for dfr in chunks:
            common.prepend_index_levels(
                dfr, SAT=SAT,
                lost_immunity_susceptibility=susceptibility
            )
            hdfstore.put(dfr)
