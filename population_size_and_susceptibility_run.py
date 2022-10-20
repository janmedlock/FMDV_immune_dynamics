#!/usr/bin/python3
'''Run simulations with varying population size and
susceptibility. This produces a file called
`population_size_and_susceptibility.h5`.'''

import numpy

import common
import h5
import population_size_and_susceptibility as psas


def _is_monotone_increasing(arr):
    return (numpy.diff(arr) > 0).all()


def _get_persistence(SAT, lost_immunity_susceptibility, population_size,
                     store, store_extinction_time):
    '''Get the proportion of simulations where the pathogen persisted
    over the whole time interval.'''
    where = ' & '.join((f'{SAT=}',
                        f'{lost_immunity_susceptibility=}',
                        f'{population_size=}'))
    extinction_time = common.get_extinction_time(store, where=where)
    store_extinction_time.put(extinction_time)
    persisted = ~extinction_time.observed
    persistence = sum(persisted) / len(persisted)
    return persistence


def _run_over_population_sizes(SAT, susceptibility, nruns,
                               store, store_extinction_time):
    '''For the given SAT and susceptibility, run simulations with
    varying population size.'''
    # Ignoring sampling error, persistence is an increasing function
    # of population size, so if persistence is 100% for some
    # population size, it will also be 100% for all larger population
    # sizes. Thus, we end this loop if persistence reaches 100%.
    for population_size in psas.population_sizes:
        psas.run(SAT, susceptibility, population_size, nruns, store)
        persistence = _get_persistence(SAT, susceptibility, population_size,
                                       store, store_extinction_time)
        print(
            f'{SAT=}, {susceptibility=}, {population_size=}: {persistence=}'
        )
        if persistence == 1.:
            break


def run(nruns):
    '''Run the simulations for the sensitivity analysis.'''
    # The logic in the innre loop `_run_over_population_sizes()`
    # requires that the population sizes be monotone increasing.
    assert _is_monotone_increasing(psas.population_sizes)
    # Extinction time is computed in the inner loop
    # `_run_over_population_sizes()` and stored to avoid having to
    # compute it again later for plotting.
    store_extinction_time_path = common.get_path_extinction_time(
        psas.store_path)
    with (h5.HDFStore(psas.store_path) as store,
          h5.HDFStore(store_extinction_time_path) as store_extinction_time):
        for SAT in common.SATs:
            for susceptibility in psas.susceptibilities:
                _run_over_population_sizes(SAT, susceptibility, nruns,
                                           store, store_extinction_time)
        store.repack()
        store_extinction_time.repack()


if __name__ == '__main__':
    NRUNS = 1000

    run(NRUNS)
