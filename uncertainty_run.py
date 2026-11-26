#!/usr/bin/env python
'''Run simulations with uncertainty in the probabilities of
lost-immunity true and recovered true. This produces a file called
`uncertainty.h5`.'''

import itertools

import joblib
import numpy
import pandas

import common
import h5
import herd
import logging_
import uncertainty


def _save_result(store, result):
    '''Save infected daily and extinction time.'''
    common.save_result(store, result,
                       infected_daily=True, extinction_time=True)


def run_one(SAT, *args, run_number=None, seed=None,
            tmax=common.TIME_MAX, print_duration=False, logging_prefix=None,
            **kwds):
    if run_number is not None:
        logging_prefix = logging_.append_to_prefix(logging_prefix,
                                                   run_number=run_number)
    rng = numpy.random.default_rng(seed)
    params = herd.Parameters(SAT=SAT,
                             sample_antibody_precisions=True,
                             seed=rng)
    herd_ = herd.Herd(params, *args, seed=rng, **kwds)
    result = herd_.run(tmax,
                       print_duration=print_duration,
                       logging_prefix=logging_prefix)
    common.prepend_index_levels(
        result,
        SAT=SAT,
        run=run_number,
        lost_immunity_true_probability=params.lost_immunity_true_probability,
        recovered_true_probability=params.recovered_true_probability,
    )
    return result


def _build_jobs(nruns, SAT, *args, seed=None, **kwds):
    logging_prefix = f'{SAT=}'
    rngs = numpy.random.default_rng(seed).spawn(nruns)
    return (
        (joblib.delayed)(run_one)(SAT, *args,
                                  run_number=run_number,
                                  seed=rng,
                                  logging_prefix=logging_prefix,
                                  **kwds)
        for (run_number, rng) in enumerate(rngs)
    )


def run_many_chunked(nruns, *args, chunksize=100, n_jobs=-1, seed=common.SEED,
                     debug=False, print_duration=None, **kwds):
    if chunksize < 1:
        chunksize = nruns
    if print_duration is None:
        print_duration = not debug
    # Use the same `seed` for each `SAT`
    jobs = itertools.chain.from_iterable(
        _build_jobs(nruns, SAT, *args,
                    seed=seed, debug=debug,
                    print_duration=print_duration, **kwds)
        for SAT in common.SATs
    )
    with joblib.Parallel(n_jobs=n_jobs, return_as='generator') as parallel:
        results = parallel(jobs)
        chunker = itertools.batched(results, chunksize)
        for chunk in chunker:
            yield pandas.concat(chunk, copy=False)


def run(nruns, *args, **kwargs):
    '''Run `nruns` for `SAT`.'''
    with h5.HDFStore(uncertainty.store_path) as store:
        chunks = run_many_chunked(nruns, *args, **kwargs)
        for chunk in chunks:
            _save_result(store, chunk)
        store.repack()
        store.set_read_only()


if __name__ == '__main__':
    # Running these requires solving for many initial conditions,
    # which is limited by the amount of memory, rather than CPUs, on
    # my machines. It uses about 0.5 GB per process.
    N_JOBS = min(22, joblib.cpu_count())
    CHUNKSIZE = 1  # Save at each completed run
    common.nice_self()
    run(common.NRUNS, n_jobs=N_JOBS, chunksize=CHUNKSIZE)
