#!/usr/bin/python3

import sys

import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters
sys.path.pop()


def check(solver):
    solver.update_blocks()
    A = solver.get_A()
    b = solver.get_b()
    newborn_proportion_immune = b[0, 0]
    n = len(numpy.arange(0, solver.age_max, solver.step)) + 1
    assert numpy.isclose(solver.params.newborn_proportion_immune,
                         newborn_proportion_immune)
    hazard_infection = ((A[n + 1, n + 1] - 1) * 2 / solver.step
                        - solver.params.hazard.mortality[0])
    assert numpy.allclose(solver.params.hazard.infection,
                          hazard_infection)


if __name__ == '__main__':
    parameters = Parameters(SAT=1)
    solver = initial_conditions.immune_status.Solver(parameters)
    check(solver)
    solver.params.newborn_proportion_immune = 0.5
    solver.params.hazard.infection = 2
    check(solver)
