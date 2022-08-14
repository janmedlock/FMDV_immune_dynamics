#!/usr/bin/python3
'''Run missing samples.'''

from joblib import delayed, Parallel

import pandas

import common
import herd
import herd.samples
import samples


def get_samples_missing(sample_number_max=None):
    '''Find the missing samples.'''
    smpls = herd.samples.load()
    missing = {}
    for SAT in common.SATs:
        idx = [num
               for num in smpls.index[:sample_number_max]
               if not samples.sample_path(SAT, num).exists()]
        missing[SAT] = smpls.loc[idx, SAT]
    return pandas.concat(missing, names=['SAT'])


def run_missing(samples_missing, *args, n_jobs=-1, **kwargs):
    '''Run `samples_missing`.'''
    prm = {SAT: herd.Parameters(SAT=SAT)
           for SAT in common.SATs}
    jobs = (delayed(samples.run_one_and_save)(prm[SAT], smpl, num,
                                              samples.sample_path(SAT, num),
                                              *args, logging_prefix=f'{SAT=}',
                                              **kwargs)
            for ((SAT, num), smpl) in samples_missing.iterrows())
    Parallel(n_jobs=n_jobs)(jobs)


if __name__ == '__main__':
    SAMPLE_NUMBER_MAX = 10300
    N_JOBS = 20

    samples_missing = get_samples_missing(SAMPLE_NUMBER_MAX)
    run_missing(samples_missing, n_jobs=N_JOBS)
