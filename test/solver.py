#!/usr/bin/python3

import sys

import matplotlib.pyplot
import numpy

sys.path.append('..')
from herd import initial_conditions, Parameters, utility
sys.path.pop()


def check_parameters(solver):
    '''Ensure the solver parameters correctly set the matrix and vector
    elements.'''
    solver.update_blocks()
    A = solver.get_A()
    b = solver.get_b()
    n = len(solver.ages)
    assert numpy.isclose(b[0, 0],
                         solver.params.newborn_proportion_immune)
    assert numpy.isclose(b[n, 0],
                         1 - solver.params.newborn_proportion_immune)
    assert numpy.allclose(A[n + 1, n + 1],
                          (1 + ((solver.params.hazard.mortality[0]
                                 + solver.params.hazard.infection)
                                * solver.step / 2)))


def plot_blocks(solver):
    A = solver.get_A()
    n = len(solver.ages)
    m = int(A.shape[0] / n)
    assert A.shape[0] == m * n
    blocks = numpy.empty((m, m), dtype=int)
    for j in range(m):
        J = slice(j * n, (j + 1) * n)
        for k in range(m):
            K = slice(k * n, (k + 1) * n)
            B = A[J, K]
            b_nonzero = B[B.nonzero()]
            if b_nonzero.shape[1] == 0:
                blocks[j, k] = 0
            elif (b_nonzero > 0).any():
                blocks[j, k] = 1
            else:
                blocks[j, k] = -1
    (fig, ax) = matplotlib.pyplot.subplots(constrained_layout=True)
    ax.matshow(blocks, cmap='PiYG')
    vars_ = solver.vars_ODE + solver.vars_PDE
    ax.set_xticks(range(len(vars_)))
    ax.set_yticks(range(len(vars_)))
    ax.set_xticklabels(vars_)
    ax.set_yticklabels(vars_)


def plot_solution(P):
    (fig, ax) = matplotlib.pyplot.subplots(constrained_layout=True)
    ages = utility.arange(0, solver.age_max, solver.step, endpoint=True)
    ax.stackplot(ages, P.T, labels=P.columns)
    ax.set_xlabel('age (y)')
    ax.set_ylabel('probability given alive')
    ax.legend()
    return ax


if __name__ == '__main__':
    parameters = Parameters(SAT=1)
    solver = initial_conditions.immune_status.Solver(parameters)
    newborn_proportion_immune = 0.6
    hazard_infection = 2
    P = solver.solve_step(solver.transform((hazard_infection,
                                            newborn_proportion_immune)))
    check_parameters(solver)
    plot_blocks(solver)
    plot_solution(P)
    matplotlib.pyplot.show()
