#!/usr/bin/python3

import os.path

import numpy

import h5
import herd
import run_common


def _copy_run_SATs(SAT, val, nruns, hdfstore_out):
    '''Copy the data from 'run_SATs.h5'.'''
    where = f'SAT={SAT} & run<{nruns}'
    with h5.HDFStore('run_SATs.h5', mode='r') as hdfstore_in:
        for chunk in hdfstore_in.select(where=where, iterator=True):
            run_common._insert_index_levels(chunk, 2,
                                            lost_immunity_susceptibility=val)
            hdfstore_out.put(chunk)


def run_susceptibility(SAT, val, tmax, nruns, hdfstore):
    if val == 1:
        _copy_run_SATs(SAT, val, nruns, hdfstore)
    else:
        p = herd.Parameters(SAT=SAT)
        p.lost_immunity_susceptibility = val
        logging_prefix = (', '.join((f'SAT {SAT}',
                                     f'lost_immunity_susceptibility {val}'))
                          + ', ')
        df = run_common.run_many(p, tmax, nruns,
                                 logging_prefix=logging_prefix)
        run_common._prepend_index_levels(df, SAT=SAT,
                                         lost_immunity_susceptibility=val)
        hdfstore.put(df)


if __name__ == '__main__':
    susceptibilities = numpy.linspace(0, 1, 11)
    nruns = 1000
    tmax = 10

    _filebase, _ = os.path.splitext(__file__)
    _filename = _filebase + '.h5'
    with h5.HDFStore(_filename) as store:
        for susceptibility in susceptibilities:
            for SAT in (1, 2, 3):
                run_susceptibility(SAT, susceptibility, tmax, nruns, store)
        store.repack()
