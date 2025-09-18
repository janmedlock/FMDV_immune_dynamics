#!/usr/bin/python3
'''Run simulations with varying population size and
susceptibility. This produces a file called
`population_size_and_susceptibility.h5`.'''

import common
import h5
import population_size as population_size_
import population_size_and_susceptibility
import susceptibility


def _build_extinction_time(store, store_extinction_time, **kwds):
    '''Get and save extinction time.'''
    store.flush(fsync=True)
    dfr = h5.load(store.filename,
                  mode='a',  # Must match mode of already open file.
                  columns=common.cols_infected,
                  use_dask=True)
    index = dfr.index.to_frame()
    mask = (
        index[list(kwds.keys())] == kwds.values()
    )
    extinction_time = common.get_extinction_time(dfr[mask])
    store_extinction_time.put(extinction_time)
    return extinction_time


def _get_persistence(store, store_extinction_time, **kwds):
    '''Get the proportion of simulations where the pathogen persisted
    over the whole time interval.'''
    extinction_time = _build_extinction_time(
        store, store_extinction_time, **kwds
    )
    return common.get_persistence(extinction_time)


def _run_over_population_sizes(SAT, lost_immunity_susceptibility, nruns,
                               store, store_extinction_time,
                               *args, **kwds):
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
            nruns, store, copy_only, *args, **kwds)
        # Calculate `persistence` if data was added to `store`.
        if stored:
            persistence = _get_persistence(
                store, store_extinction_time,
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
    assert common.is_increasing(population_size_.values, strict=True)
    # Extinction time is computed in the inner loop
    # `_run_over_population_sizes()` and stored to avoid having to
    # compute it again later for plotting.
    store_path = population_size_and_susceptibility.store_path
    store_extinction_time_path = common.get_path_extinction_time(
        population_size_and_susceptibility.store_path)
    with (h5.HDFStore(store_path) as store,
          h5.HDFStore(store_extinction_time_path) as store_extinction_time):
        for SAT in common.SATs:
            for suscept in susceptibility.values:
                _run_over_population_sizes(SAT, suscept, nruns,
                                           store, store_extinction_time,
                                           *args, **kwds)
        store.repack()
        store_extinction_time.repack()
    common.set_read_only(store_path)


if __name__ == '__main__':
    common.nice_self()
    run(common.NRUNS)
