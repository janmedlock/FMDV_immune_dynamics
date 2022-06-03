#!/usr/bin/python3

import itertools

from joblib import delayed, Parallel

import herd
import run
import susceptibility_run


def cache_seed(SAT, **kwds):
    msg = ', '.join(itertools.chain(
        (f'{SAT=}',),
        (f'{key}={val}' for (key, val) in kwds.items()),
    ))
    print(msg)
    parameters = herd.Parameters(SAT=SAT)
    for (key, val) in kwds.items():
        assert hasattr(parameters, key)
        setattr(parameters, key, val)
    rvs = herd.RandomVariables(parameters)


if __name__ == '__main__':
    n_jobs = 1

    jobs = itertools.chain(
        (delayed(cache_seed)(SAT)
         for SAT in run.SATs),
        (delayed(cache_seed)(SAT,
                             lost_immunity_susceptibility=susceptibility)
         for susceptibility in susceptibility_run.susceptibilities
         for SAT in run.SATs),
    )

    Parallel(n_jobs=n_jobs)(jobs)
