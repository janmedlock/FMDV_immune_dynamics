#!/usr/bin/python3
from context import herd
from herd.initial_conditions import immune_status


if __name__ == '__main__':
    for SAT in (1, 2, 3):
        print(f'SAT{SAT}')
        parameters = herd.Parameters(SAT=SAT)
        solver = immune_status.Solver(parameters)
        P = immune_status.probability_interpolant(parameters)._probability
        print('hazard_infection = {}'.format(
            solver.get_hazard_infection(P)))
        print('newborn_proportion_immune = {}'.format(
            solver.get_newborn_proportion_immune(P)))
