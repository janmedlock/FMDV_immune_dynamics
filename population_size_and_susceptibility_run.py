#!/usr/bin/python3
'''Run simulations with varying population size and
susceptibility. This produces a file called
`population_size_and_susceptibility.h5`.'''

import numpy

import common
import h5
import herd.utility
import population_size as population_size_
import population_size_and_susceptibility
import susceptibility


def _get_persistence(store, **parameters):
    '''Get the proportion of simulations where the pathogen persisted
    over the whole time interval for the given parameter values.'''
    extinction_time = store.select('extinction_time')
    mask = numpy.all(
        [
            extinction_time.index.get_level_values(level) == value
            for (level, value) in parameters.items()
        ],
        axis=0
    )
    return common.get_persistence(extinction_time[mask])


def _run_over_population_sizes(SAT, lost_immunity_susceptibility, nruns,
                               store, *args, **kwds):
    '''For the given SAT and susceptibility, run simulations with
    varying population size.'''
    # Ignoring sampling error, persistence is an increasing function
    # of population size, so if persistence is 100% for some
    # population size, it will also be 100% for all larger population
    # sizes. Thus, we won't run simulations for larger population
    # sizes once we have found a population size with 100%
    # persistence. But if we have some already run simulations for
    # larger population sizes from the 1-parameter sensitivity runs of
    # population size and susceptibility, do add those to the
    # output. If `copy_only` is `True`, new simulations are not run,
    # but the 1-parameter sensitivity runs are still copied.
    copy_only = False
    for population_size in population_size_.values:
        stored = population_size_and_susceptibility.run(
            SAT, lost_immunity_susceptibility, population_size,
            nruns, store, copy_only, *args, **kwds
        )
        # Calculate `persistence` if data was added to `store`.
        if stored:
            persistence = _get_persistence(
                store,
                SAT=SAT,
                lost_immunity_susceptibility=lost_immunity_susceptibility,
                population_size=population_size,
            )
            print(', '.join((f'{SAT=}',
                             f'{lost_immunity_susceptibility=!s}',
                             f'{population_size=!s}'))
                  + f': {persistence=}')
            if persistence == 1.:
                copy_only = True


def run(nruns, *args, **kwds):
    '''Run the simulations for the sensitivity analysis.'''
    # The logic in the inner loop `_run_over_population_sizes()`
    # requires that the population sizes be strictly increasing.
    assert herd.utility.is_increasing(population_size_.values, strict=True)
    with h5.HDFStore(population_size_and_susceptibility.store_path) as store:
        for SAT in common.SATs:
            for suscept in susceptibility.values:
                _run_over_population_sizes(SAT, suscept, nruns, store,
                                           *args, **kwds)
        store.repack()
        store.set_read_only()


if __name__ == '__main__':
    common.nice_self()
    run(common.NRUNS)
