#!/usr/bin/python3

import sys

sys.path.append('..')
from herd import Parameters
from herd.initial_conditions import immune_status
sys.path.pop()


if __name__ == '__main__':
    for SAT in (1, 2, 3):
        print(f'SAT {SAT}')
        parameters = Parameters(SAT=SAT)
        solver = immune_status.Solver(parameters)
        P = immune_status.probability_interpolant(parameters)._probability
        print('hazard_infection = {}'.format(
            solver.get_hazard_infection(P)))
        print('newborn_proportion_immune = {}'.format(
            solver.get_newborn_proportion_immune(P)))
