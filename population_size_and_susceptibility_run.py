#!/usr/bin/python3
'''Run simulations with varying population size and
susceptibility. This produces a file called
`population_size_and_susceptibility.h5`.'''

import common
import h5
import population_size_and_susceptibility as psas


def get_persistence(SAT, lost_immunity_susceptibility, population_size,
                    hdfstore):
    '''Get the proportion of simulations where the pathogen persisted
    over the whole time interval.'''
    where = ' & '.join((f'{SAT=}',
                        f'{lost_immunity_susceptibility=}',
                        f'{population_size=}'))
    extinction_time = common.get_extinction_time(hdfstore, where=where)
    persisted = ~extinction_time.observed
    persistence = sum(persisted) / len(persisted)
    return persistence


if __name__ == '__main__':
    NRUNS = 1000

    with h5.HDFStore(psas.store_path) as store:
        for SAT in common.SATs:
            for susceptibility in psas.susceptibilities:
                for population_size in psas.population_sizes:
                    psas.run(SAT, susceptibility, population_size,
                             NRUNS, store)
                    persistence = get_persistence(SAT, susceptibility,
                                                  population_size, store)
                    print(f'{SAT=}, {susceptibility=}, {population_size=}'
                          f': {persistence=}')
                    if persistence == 1.:
                        # (Up to sampling error) persistence is 100%
                        # for the current `population_size`, which
                        # will also be true for larger population
                        # sizes, so don't run them and just skip ahead
                        # to the next value of susceptibility & SAT.
                        break
        store.repack()
