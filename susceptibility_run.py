#!/usr/bin/python3

import pathlib

import numpy

import h5
import herd
import run


store_path = pathlib.Path(__file__).with_suffix('.h5')

susceptibilities = numpy.linspace(0, 1, 11)


def _copy_run(SAT, susceptibility, nruns, hdfstore_out):
    '''Copy the data from 'run.h5'.'''
    where = f'SAT={SAT} & run<{nruns}'
    with h5.HDFStore(run.store_path, mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run.insert_index_levels(
                chunk, 2,
                lost_immunity_susceptibility=susceptibility
            )
            hdfstore_out.put(chunk)


def run_susceptibility(SAT, susceptibility, tmax, nruns, hdfstore, *args,
                       chunksize=-1, n_jobs=-1, **kwargs):
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
        chunks = run.run_many_chunked(parameters, tmax, nruns, *args,
                                      chunksize=chunksize, n_jobs=n_jobs,
                                      logging_prefix=logging_prefix, **kwargs)
        for dfr in chunks:
            run.prepend_index_levels(
                dfr, SAT=SAT,
                lost_immunity_susceptibility=susceptibility
            )
            hdfstore.put(dfr)


if __name__ == '__main__':
    tmax = 10
    nruns = 1000
    chunksize = 100
    n_jobs = -1

    with h5.HDFStore(store_path) as store:
        for SAT in run.SATs:
            for susceptibility in susceptibilities:
                run_susceptibility(SAT, susceptibility, tmax, nruns, store,
                                   chunksize=chunksize, n_jobs=n_jobs)
        store.repack()
