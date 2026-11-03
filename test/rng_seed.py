#!/usr/bin/env python
'''Test reproducibility of the random-number generator.'''

import concurrent.futures
import functools
import itertools
import os

import numpy

from context import common, herd


def get_rng(herd_):
    # Collect all of the `.rng` attributes inside `.herd_`,
    # check that they are all the same object,
    # and return that object
    rvs = herd_.rvs
    rngs_rvs = (getattr(rvs, rv).rng for rv in vars(rvs))
    rng_events = herd_.events.rng
    rngs = itertools.chain(rngs_rvs, (rng_events,))
    rng = next(rngs)
    assert all(r is rng for r in rngs)
    return rng


def get_result_and_rng(parameters, run_number=None, tmax=common.TIME_MAX,
                       **kwds):
    run_str = '' if run_number is None else f' #{run_number}'
    print(f'Running `herd.Herd().run()`{run_str}')
    herd_ = herd.Herd(parameters, **kwds)
    result = herd_.run(tmax)
    print(f'Finished `herd.Herd().run()`{run_str}')
    return (result, get_rng(herd_))


def get_results_and_rngs(nruns, seed=common.SEED, **kwds):
    # Build `herd_.RandomVariables()` once to initialize the cache
    rng = numpy.random.default_rng(seed)
    parameters = herd.Parameters(seed=rng, **kwds)
    herd.RandomVariables(parameters)
    get_result_and_rng_ = functools.partial(get_result_and_rng,
                                            parameters, seed=rng)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        results_and_rngs = pool.map(get_result_and_rng_, range(nruns))
        return zip(*results_and_rngs)


def check_results(results):
    assert all(a.equals(b) for (a, b) in itertools.pairwise(results))
    print('`results` are equal')


def check_rngs(rngs):
    assert not any(a is b for (a, b) in itertools.pairwise(rngs))
    print('`rngs` are different objects')
    randoms = (rng.random() for rng in rngs)
    assert all(x == y for (x, y) in itertools.pairwise(randoms))
    print('`rngs` give same `.random()` value')


if __name__ == '__main__':
    SAT = 1
    nruns = max(os.process_cpu_count(), 2)
    (results, rngs) = get_results_and_rngs(nruns, SAT=SAT)
    check_results(results)
    check_rngs(rngs)
