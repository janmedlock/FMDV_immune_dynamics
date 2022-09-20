#!/usr/bin/python3
'''Re-run samples with empty data files.'''

from joblib import delayed, Parallel

import pandas

import common
import herd
import herd.samples
import samples


def get_samples_empty(sample_number_max=None):
    '''Find the empty samples.'''
    smpls = herd.samples.load()
    empty = {}
    for SAT in common.SATs:
        idx = [num
               for num in smpls.index[:sample_number_max]
               if not samples.sample_path(SAT, num).exists()]
        empty[SAT] = smpls.loc[idx, SAT]
    return pandas.concat(empty, names=['SAT'])


def run_empty(samples_empty, *args, n_jobs=-1, **kwargs):
    '''Run `samples_empty`.'''
    prm = {SAT: herd.Parameters(SAT=SAT)
           for SAT in common.SATs}
    jobs = (delayed(samples.run_one_and_save)(prm[SAT], smpl, num,
                                              samples.sample_path(SAT, num),
                                              *args, logging_prefix=f'{SAT=}',
                                              **kwargs)
            for ((SAT, num), smpl) in samples_empty.iterrows())
    Parallel(n_jobs=n_jobs)(jobs)


if __name__ == '__main__':
    SAMPLE_NUMBER_MAX = 10300
    N_JOBS = 20

    samples_empty = get_samples_empty(SAMPLE_NUMBER_MAX)
    run_empty(samples_empty, n_jobs=N_JOBS)
