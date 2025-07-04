#!/usr/bin/python3
'''Show the hazard of infection and the proportion of newborns immune
at the endemic equilibrium.'''

from context import common
from context import herd
from herd.initial_conditions.immune_status.solver import get_equilibrium


def get_equilibria():
    return {
        SAT: get_equilibrium(herd.Parameters(SAT=SAT))
        for SAT in common.SATs
    }


if __name__ == '__main__':
    equilibria = get_equilibria()
    for (SAT, equilibrium) in equilibria.items():
        print(f'SAT{SAT}')
        for (key, val) in equilibrium.items():
            print(f'{key}={val}')
