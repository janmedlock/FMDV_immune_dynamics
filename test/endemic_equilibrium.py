#!/usr/bin/env python
'''Show the hazard of infection and the proportion of newborns immune
at the endemic equilibrium.'''

from context import common, herd
import herd._initial_conditions.immune_status.solver


def get_equilibria():
    return {
        SAT: herd._initial_conditions.immune_status.solver.get_equilibrium(
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
