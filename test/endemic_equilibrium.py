#!/usr/bin/env python
'''Show the hazard of infection and the proportion of newborns immune
at the endemic equilibrium.'''

from _context import common, herd
import herd.initial_conditions.test


def get_equilibria():
    return {
        SAT: herd.initial_conditions.test.get_equilibrium(
            herd.Parameters(SAT=SAT)
        )
        for SAT in common.SATs
    }


if __name__ == '__main__':
    equilibria = get_equilibria()
    for (SAT, equilibrium) in equilibria.items():
        print(f'SAT{SAT}')
        for (key, val) in equilibrium.items():
            print(f'{key}={val}')
